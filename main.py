import json
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from groq import Groq

from db import init_db, db_save_note, db_search_notes, db_save_message, db_get_messages

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

# 2. A Dictonary that maps tool names to the actual functions
available_tools = {
    "get_current_time": get_current_time,
    "save_note": save_note,
    "search_notes": search_notes,
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
    }
]

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. Fetch recent history from DB and save the new user message
    history = db_get_messages(req.session_id, limit=10)
    user_turn = {"role": "user", "content": req.message}
    db_save_message(req.session_id, user_turn)

    # 2. Build the messages list: System prompt + DB History + Current User turn
    messages = [
        {"role": "system", "content": "You are a helpful personal assistant."}
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

    