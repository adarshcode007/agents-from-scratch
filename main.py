import json
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from groq import Groq
from embeddings import get_embedding
from typing import Literal
from uuid import uuid4

from db import (
    init_db, db_save_note, db_search_notes, db_save_message,
    db_get_messages, db_save_memory, db_get_memories, db_search_memories,
    db_add_task, db_list_tasks, db_complete_task,
    db_get_session_summary, db_update_session_summary, db_get_message_count,
    db_delete_task, db_create_pending_action, db_get_pending_action, db_update_action_status
)

# Load .env file
load_dotenv()
init_db()

# Initialize FastAPI and Groq
app = FastAPI(title="Assistant Agent")
client = Groq()

# 1. The actual Python function
def get_current_time():
    """ Returns the current date and time as a readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_note(text: str):
    """Saves the response to the notes list"""
    db_save_note(text)
    return f"Note saved: {text}"

def search_notes(query: str):
    """Search the notes by its keyword."""
    results = db_search_notes(query)
    if results:
        return f"Note found: {', '.join(results)}"
    return f"No notes found with the keyword {query}"

def remember(fact: str):
    """Remembers the fact requested by the user"""
    vector = get_embedding(fact)
    return db_save_memory(fact, embedding=vector)

def add_task(title: str, due_date: str = None, description: str = None):
    """Add a new task with optional due date and description."""
    return db_add_task(title, due_date, description)

def list_tasks(status: str = "pending"):
    """List tasks filtered by status ('pending' or 'done')."""
    tasks = db_list_tasks(status)
    if not tasks:
        return f"No {status} tasks found."
    
    # Format into a clean string for LLM
    formatted = [
        f"[ID: {t['id']}] {t['title']}" + 
        (f" (Due: {t['due_date']})" if t['due_date'] else "") + 
        (f" - {t['description']}" if t.get('description') else "")
        for t in tasks
    ]
    return f"Tasks ({status}):\n" + "\n".join(formatted)

def complete_task(task_id: int):
    """Mark a specific task ID as done."""
    return db_complete_task(task_id)

def delete_task(task_id: int):
    """Permanently delte a task by ID."""
    return db_delete_task(task_id)

def compact_session_history(session_id: str):
    """Summarizes older conversation turns if message count exceeds threshold."""
    count = db_get_message_count(session_id)

    # Only run when count is greater than 10, then once every 10 messages to prevent over-summarizing
    if count >= 10 and count % 10 == 0:
        old_summary = db_get_session_summary(session_id)
        all_messages = db_get_messages(session_id, limit=30)
        messages_text = "\n".join([
            f"{m['role'].upper()}: {m.get('content', '[Tool Call]')}"
            for m in all_messages if m.get('content')
        ])
        summary_prompt = f"""
            You are an expert conversation summarizer.

            Current existing summary:
            "{old_summary}"

            New conversation messages to incorporate:
            {messages_text}

            Task: Provide an updated, concise, high-density summary of the conversation so far.
            Keep key decisions, user constraints, tasks mentioned, and personal context.
            Return ONLY the summary paragraph.
        """

        # Call LLM to generate the summary
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": summary_prompt}
            ]
        )
        new_summary = response.choices[0].message.content

        # Save updated summary into SQLite
        db_update_session_summary(session_id, new_summary)

# 2. A Dictonary that maps tool names to the actual functions
available_tools = {
    "get_current_time": get_current_time,
    "save_note": save_note,
    "search_notes": search_notes,
    "remember": remember,
    "add_task": add_task,
    "list_tasks": list_tasks,
    "complete_task": complete_task,
    "delete_task": delete_task,
}

# High-Risk Tools that require human approval:
HIGH_RISK_TOOLS = {"delete_task"}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time. Use this whenever the user asks about the time, date, day, or current moment.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },{
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note or memo for the user when they ask to record or remember something.",
            "parameters":{
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note to save."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search for notes by keyword when they ask to search for a note with a keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":{
                        "type": "string",
                        "description": "The keyword to search for in the notes."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Remember the facts given by the user when they explicitly ask to remember.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to be remembered."
                    }
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a new task with optional due date and description when they ask to add a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the task."
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM). Use get_current_time first if you need to calculate relative dates like 'tomorrow' or 'next Monday'."
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of the task."
                    }
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List tasks filtered by status ('pending' or 'done').",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "The status of the task."
                    }
                },
                "required": ["status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a specific task ID as done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to complete."
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": "Permanently delete a task from the database by it ID. CAUTION: This is an irreversible destructive action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to permanently delete."
                    }
                },
                "required": ["task_id"]
            }
        }
    }
]

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class ChatResponse(BaseModel):
    response: str

class PlannedStep(BaseModel):
    step_number: int
    title: str
    description: str
    estimated_minute: int
    priority: Literal["low", "medium", "high"]

class ActionPlan(BaseModel):
    goal: str
    summary: str
    steps: list[PlannedStep]
    total_estimated_minutes: int

class PlanRequest(BaseModel):
    goal: str

# For Plan-and-Solve
class StepExecutionResult(BaseModel):
    step_number: int
    title: str
    status: str # "completed" | "failed"
    tool_action: str

class PlanExecutionResponse(BaseModel):
    goal: str
    plan_summary: str
    total_steps: int
    executed_steps: list[StepExecutionResult]
    final_message: str

# For Reflect and Generate
class ReflectionRequest(BaseModel):
    task: str
    criteria: list[str]

class CritiqueResult(BaseModel):
    passed: bool
    critique: str
    issues_found: list[str]

class ReflectionResponse(BaseModel):
    task: str
    initial_draft: str
    critique: CritiqueResult
    final_output: str
    refined: bool

# For Human-in-the-loop
class ApprovalRequest(BaseModel):
    action_id: str
    decision: str   # "approved" or "rejected"

class ApprovalResponse(BaseModel):
    action_id: str
    status: str
    result: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    # Semantic vector search for relevant memories
    query_vector = get_embedding(req.message)
    relevant_memories = db_search_memories(query_vector, top_k=3, threshold=0.3)

    summary = db_get_session_summary(req.session_id)

    system_content = "You are a helpful personal assistant."

    if relevant_memories:
        system_content += "\n\nKnown facts about the user:\n" + "\n".join(["-" + m for m in relevant_memories])
    
    if summary:
        system_content += f"\n\nSummary of earlier conversation:\n{summary}"
    
    # 1. Fetch recent history from DB and save the new user message
    history = db_get_messages(req.session_id, limit=6)
    user_turn = {"role": "user", "content": req.message}
    db_save_message(req.session_id, user_turn)

    # 2. Build the messages list: System prompt + DB History + Current User turn
    messages = [
        {"role": "system", "content": system_content}
    ] + history + [user_turn]

    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        # 3. Call the LLM with available tools
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools_schema,
        )

        response_message = completion.choices[0].message

        # 4. Save the assistant response to DB and local loop history
        db_save_message(req.session_id, {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": response_message.tool_calls
        })
        messages.append(response_message)

        # 5. If the model did not call any tools, queue background compaction and return final text
        if not response_message.tool_calls:
            background_tasks.add_task(compact_session_history, req.session_id)
            return ChatResponse(response=response_message.content or "")
        
        # 6. If tools were called, execute each one
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            
            # 6a. Defensively parse arguments
            try:
                tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            except json.JSONDecodeError:
                tool_result = f"Error: Invalid JSON arguments provided for tool '{tool_name}'."
            else:
                # HITL CHECK: Intercept high-risk tools before execution
                if tool_name in HIGH_RISK_TOOLS:
                    action_id = str(uuid4())
                    db_create_pending_action(action_id, req.session_id, tool_name, tool_args, tool_call_id=tool_call.id)

                    # Persist the tool message stating execution is held for approval
                    tool_turn = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": f"[PAUSED] Tool '{tool_name}' is high-risk and pending human approval (action_id: {action_id})."
                    }
                    db_save_message(req.session_id, tool_turn)

                    # Return immediately to the user with the action_id
                    return ChatResponse(
                        response=(
                            f"⚠️ **Approval Required**: The model requested to run high-risk tool `{tool_name}`.\n\n"
                            f"- **Action ID**: `{action_id}`\n"
                            f"- **Parameters**: `{json.dumps(tool_args)}`\n\n"
                            f"To execute or reject, send a request to `POST /approve` with this `action_id`."
                        )
                    )

                # 6b. Defensively execute tool
                if tool_name in available_tools:
                    tool_func = available_tools[tool_name]
                    try:
                        tool_result = tool_func(**tool_args)
                    except TypeError as e:
                        tool_result = f"Error: Invalid parameters for '{tool_name}'. Details: {e}"
                    except Exception as e:
                        tool_result = f"Error executing '{tool_name}': {e}"
                else:
                    tool_result = f"Error: Tool '{tool_name}' does not exist."

            # 7. Build tool message, persist to DB, and append to local loop
            tool_turn = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(tool_result)
            }
            db_save_message(req.session_id, tool_turn)
            messages.append(tool_turn)
        
        iteration += 1
    
    background_tasks.add_task(compact_session_history, req.session_id)
    raise HTTPException(status_code=500, detail="Agent loop exceeded maximum iterations")

@app.post("/plan", response_model=ActionPlan)
def generate_plan(req: PlanRequest):
    """Generate a structured, decomposed action plan from a high-level goal."""

    # 1. Provide the exact JSON Schema in the system prompt
    schema_json = ActionPlan.model_json_schema()

    prompt = f"""You are a master project planner and decomposition engine.
                Decompose the user's goal into logical, sequential, actionable steps.
                You MUST reply with a JSON object conforming strictly to this JSON Schema:
                {json.dumps(schema_json, indent=2)}
                """
    
    # 2. Call the LLM with JSON Mode enabled
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": req.goal}],
        response_format={"type": "json_object"}
    )

    # 3. Parse and validate through Pydantic
    raw_json = completion.choices[0].message.content
    try:
        validated_plan = ActionPlan.model_validate_json(raw_json)
        return validated_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Validate LLM output: {e}")

@app.post("/plan-and-execute", response_model=PlanExecutionResponse)
def plan_and_execute(req: PlanRequest):
    """
    Plan-and-Solve Pattern:
    1. Plan: Decompose goal into typed Pydantic ActionPlan.
    2. Solve: Iterate through planned steps, creating tasks/notes in SQLite.
    3. Synthesize: Return structured execution results.
    """
    # 1. Generate the Plan
    plan = generate_plan(req)

    executed_results = []

    # 2. Execute each step
    for step in plan.steps:
        # Automatically register each step as an actionable task in SQLite
        tool_res = db_add_task(
            title=step.title,
            due_date=None,
            description=f"Priority: {step.priority} | Est: {step.estimated_minute}m - {step.description}"
        )

        executed_results.append(StepExecutionResult(
            step_number=step.step_number,
            title=step.title,
            status="completed",
            tool_action=tool_res
        ))
    
    db_save_note(f"Plan created for goal: '{plan.goal}'. Total steps: {len(plan.steps)}")

    return PlanExecutionResponse(
        goal=plan.goal,
        plan_summary=plan.summary,
        total_steps=len(plan.steps),
        executed_steps=executed_results,
        final_message=f"Successfully decomposed and created {len(plan.steps)} actionable tasks in your database."
    )

@app.post("/reflect-and-generate", response_model=ReflectionResponse)
def reflect_and_generate(req: ReflectionRequest):
    """
    Reflection / Evaluator-Optimizer Loop:
    1. Generator: Drafts initial response.
    2. Critic: Evaluates draft against criteria using Structured Outputs.
    3. Refiner: If critic finds issues, revises the draft before returning.
    """
    criteria_list = "\n".join(f"- {c}" for c in req.criteria)

    # 1. GENERATOR: Create initial draft
    gen_resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": f"You are a specialized content creator. Complete the user's task strictly adhering to these criteria:\n{criteria_list}"},
            {"role": "user", "content": req.task}
        ]
    )
    initial_draft = gen_resp.choices[0].message.content

    # 2. CRITIC: Stirctly audit the draft against criteria using Strucutred JSON
    critic_schema = CritiqueResult.model_json_schema()
    critic_prompt = f"""
        You are a ruthless quality control evaluator and compilance auditor.
        Audit the following draft against the required criteria.

        CRITERIA: {criteria_list}
        DRAFT TO AUDIT: \"\"\"{initial_draft}\"\"\"

        Reply with a JSON object confirming stricly to this JSON Schema: {json.dumps(critic_schema, indent=2)}
    """
    critic_resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages= [
            {"role": "user", "content": critic_prompt}
        ],
        response_format={"type": "json_object"}
    )

    critique = CritiqueResult.model_validate_json(critic_resp.choices[0].message.content)

    # 3. REFINER: If failed, rewrite the draft addressing all critique points
    if not critique.passed:
        issues_text = "\n".join(f"- {issue}" for issue in critique.issues_found)
        refiner_prompt = f"""
            You are a master editor. The previous draft was rejected by the QA Critic.

            ORIGINAL TASK: {req.task}
            CRITIQUE & ISSUES TO FIX: {issues_text}

            PREVIOUS DRAFT: \"\"\"{initial_draft}\"\"\"

            Task: Rewrite the draf to completely fix every issue while satisfying all criteria.
            Return ONLY the perfected final draft.
        """

        refined_resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": refiner_prompt}]
        )
        final_output = refined_resp.choices[0].message.content
        refined = True
    else:
        final_output = initial_draft
        refined = False
    
    return ReflectionResponse(
        task=req.task,
        initial_draft=initial_draft,
        critique=critique,
        final_output=final_output,
        refined=refined
    )

@app.post("/approve", response_model=ApprovalResponse)
def handle_approval(req: ApprovalRequest):
    """Approve or reject a pending high-risk tool execution."""
    action = db_get_pending_action(req.action_id)
    if not action:
        raise HTTPException(status_code=404, detail="Action ID not found.")
    
    if action["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Action is already {action['status']}.")

    if req.decision == "approved":
        tool_name = action["tool_name"]
        tool_args = json.loads(action["tool_args"]) if action["tool_args"] else {}
        # Execute the tool
        if tool_name in available_tools:
            try:
                exec_result = available_tools[tool_name](**tool_args)
                status = "approved"
            except Exception as e:
                exec_result = f"Error executing approved tool: {e}"
                status = "failed"
        else:
            exec_result = f"Error: Tool '{tool_name}' no longer available."
            status = "failed"
    else:
        exec_result = "Action rejected by human operator."
        status = "rejected"
    
    # Update action in DB
    db_update_action_status(req.action_id, status)

    # Save resolution note in conversation history
    db_save_message(action["session_id"],{
        "role": "user",
        "content": f"[HUMAN {status.upper()}]: Action {req.action_id} was {status}. Result: {exec_result}"
    })

    return ApprovalResponse(
        action_id=req.action_id,
        status=status,
        result=str(exec_result)
    )

@app.get("/notes", response_model=list[str])
def get_all_notes():
    return db_search_notes("")

@app.get("/memories", response_model=list[str])
def get_all_memories():
    return db_get_memories()

@app.get("/tasks", response_model = dict)
def get_all_tasks():
    pending_tasks = db_list_tasks('pending')
    done_tasks = db_list_tasks('done')
    return {"pending": pending_tasks, "done": done_tasks}

@app.get("/summary/{session_id}")
def get_session_summary(session_id: str):
    """Inspect the current rolling summary for a session."""
    return {"session_id": session_id, "summary": db_get_session_summary(session_id)}

@app.post("/memory")
def get_relevant_memories(req: ChatRequest):
    query_vector = get_embedding(req.message)
    return db_search_memories(query_vector, top_k=3, threshold=0.3)