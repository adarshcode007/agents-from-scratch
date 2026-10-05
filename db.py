import sqlite3
import json

DB_FILE = "assistant.db"

def get_connection():
    """Helper to open a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row    # Allows accessing columns by name like dict
    return conn

def init_db():
    """Create the required tables if they don't already exist."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Table 1: Notes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: Messages (Chat History per session)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def db_save_note(text: str) -> str:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO notes (text) VALUES (?)", (text,))
        conn.commit()
        return f"Note saved successfully: '{text}'"

def db_search_notes(query: str) -> list[str]:
    with get_connection as conn:
        cursor = conn.cursor()
        if query:
            cursor.execute("SELECT * FROM notes WHERE text like ? ORDER BY id DESC", (f"%{query}%",))
        else:
            cursor.execute("SELECT text FROM notes ORDER BY id DESC")
            rows = cursor.fetchall()
            return [row["text"] for row in rows]

def db_save_message(session_id: str, message: dict):
    """Saves a message dictionary into the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # If the message has tool_calls objects, convert them to JSON string
        tool_calls_json = json.dumps([
            tc if isinstance(tc, dict) else tc.model_dump() for tc in message["tool_calls"]
        ])

        cursor.execute("""
            INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, name)
            VALUES(?, ?, ?, ?, ?, ?)
        """,(
            session_id, message.get("role"), message.get("content"), tool_calls_json, message.get("tool_call_id"), message.get("name")
        ))
        conn.commit()


def db_get_messages(session_id: str, limit: int = 10) -> list[dict]:
    """Retrieves the last N messages for a session in chronological order."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Fetch latest N messages (descending), then reverse them to restore chronological order
        cursor.execute("""
            SELECT role, content, tool_calls, tool_call_id, name
            FROM messages WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        """,(session_id, limit))
        rows = cursor.fetchall()
    
    messages = []
    for row in reversed(rows):
        msg = {"role": row["role"]}
        if row["content"] is not None:
            msg["content"] = row["content"]
        if row["tool_calls"]:
            msg["tool_calls"] = json.loads(row["tool_calls"])
        if row["tool_call_id"]:
            msg["tool_call_id"] = row["tool_call_id"]
        if row["name"]:
            msg["name"] = row["name"]
        messages.append(msg)
        
    return messages
