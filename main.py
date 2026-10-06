import json
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from groq import Groq

from db import (
    init_db, db_save_note, db_search_notes, db_save_message,
    db_get_messages, db_save_memory, db_get_memories,
    db_add_task, db_list_tasks, db_complete_task
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
    return db_save_memory(fact)

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

# 2. A Dictonary that maps tool names to the actual functions
available_tools = {
    "get_current_time": get_current_time,
    "save_note": save_note,
    "search_notes": search_notes,
    "remember": remember,
    "add_task": add_task,
    "list_tasks": list_tasks,
    "complete_task": complete_task,
}

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
    }
]

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    memories = db_get_memories()
    system_content = "You are a helpful personal assistant."

    if memories:
        system_content += "\n\nKnown facts about the user:\n" + "\n".join(["-" + m for m in memories])
    
    # 1. Fetch recent history from DB and save the new user message
    history = db_get_messages(req.session_id, limit=10)
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

        # 5. If the model did not call any tools, return final text
        if not response_message.tool_calls:
            return ChatResponse(response=response_message.content or "")
        
        # 6. If tools were called, execute each one
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            if tool_name in available_tools:
                tool_func = available_tools[tool_name]
                tool_result = tool_func(**tool_args)
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
    raise HTTPException(status_code=500, detail="Agent loop exceeded maximum iterations")


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