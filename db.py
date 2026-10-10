import sqlite3
import json

from embeddings import cosine_similarity

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

        # Table 3: Memories (Facts to be remembered)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        except sqlite3.OperationalError:
            pass # column already exists

        # Table 4: Tasks ()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                due_date TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 5: Sessions (stores the rolling summary for each session)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                summary TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)

        # Table 6: Pending Actions (for Human-in-the-Loop)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_actions (
                action_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_args TEXT NOT NULL,
                tool_call_id TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 7: Traces & Spans (Telemetry & Observability)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                session_id TEXT,
                step_type TEXT NOT NULL,  -- 'llm_call', 'tool_execution', 'handoff'
                name TEXT NOT NULL,   -- agent name, tool name, or model name
                input_data TEXT,
                output_data TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                latency_ms REAL NOT NULL,
                status TEXT DEFAULT 'success',  -- 'success', 'error'
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
    with get_connection() as conn:
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
        tool_calls_json = None
        if message.get("tool_calls"):
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

def db_save_memory(fact: str, embedding: list[float] = None) -> str:
    embedding_json = json.dumps(embedding) if embedding else None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (fact, embedding) VALUES (?, ?)
        """,(fact, embedding_json))
        conn.commit()
    return f"Remembered: {fact}"

def db_search_memories(query_embedding: list[float], top_k: int = 3, threshold: float = 0.3) -> list[str]:
    """Find the top-K most semantically relevant memories using cosine similarity."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT fact, embedding FROM memories WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
    
    scored_memories = []
    for row in rows:
        if row["embedding"]:
            mem_vec = json.loads(row["embedding"])
            similarity = cosine_similarity(query_embedding, mem_vec)
            if similarity >= threshold:
                scored_memories.append((similarity, row["fact"]))
    
    # sort by highest similarity score first
    scored_memories.sort(key=lambda x: x[0], reverse=True)

    # Return only the top-K facts
    return [fact for score, fact in scored_memories[:top_k]]

def db_get_memories() -> list[str]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fact FROM memories
        """)
        rows = cursor.fetchall()
    return [row["fact"] for row in rows]


def db_add_task(title: str, due_date: str = None, description: str = None) -> str:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tasks (title, due_date, description)
            VALUES (?, ?, ?)
        """,(title, due_date, description))
        conn.commit()
    return f"Task added: {title}"

def db_list_tasks(status: str = "pending") -> list[dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, due_date, status, description from tasks
            WHERE status = ?
            ORDER BY due_date 
        """,(status,))
        rows = cursor.fetchall()
    return [dict(row) for row in rows]

def db_complete_task(task_id: int) -> str:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE tasks SET status = 'done'
            WHERE id = ?
        """,(task_id,))
        conn.commit()

        # Check if any row was updated
        if cursor.rowcount == 0:
            return f"Error: Task with ID {task_id} not found."
    return f"Task {task_id} completed."

def db_get_session_summary(session_id: str) -> str:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT summary FROM sessions
            WHERE session_id = ?
        """,(session_id,))
        row = cursor.fetchall()

        if not row:
            return ""
    return row[0]["summary"]

def db_update_session_summary(session_id: str, summary: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, summary)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary = excluded.summary,
                created_at = CURRENT_TIMESTAMP
        """,(session_id, summary))
        conn.commit()

def db_get_message_count(session_id: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count FROM messages WHERE session_id = ?
        """,(session_id,))
        row = cursor.fetchone()
        return row["count"] if row else 0

# For Human-in-the-loop tool
def db_delete_task(task_id: int) -> str:
    """Permanently delete a task from SQLite."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE id = ?",(task_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return f"Error: Task with ID {task_id} not found."
        return f"Task {task_id} permanently deleted."

def db_create_pending_action(action_id: str, session_id: str, tool_name: str, tool_args: dict, tool_call_id: str = "", description: str = ""):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pending_actions (action_id, session_id, tool_name, tool_args, tool_call_id, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """,(action_id, session_id, tool_name, json.dumps(tool_args), tool_call_id, description))
        conn.commit()
    
def db_get_pending_action(action_id: str) -> dict:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM pending_actions WHERE action_id = ?
        """,(action_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def db_update_action_status(action_id: str, status: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE pending_actions SET status = ? WHERE action_id = ?
        """, (status, action_id))
        conn.commit()
    
def db_log_trace(
    trace_id: str,
    step_type: str,
    name: str,
    latency_ms: float,
    session_id: str = None,
    input_data: dict | str = None,
    output_data: dict | str = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    status: str = "success"
):
    """Logs a single execution span into SQlite traces."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO traces (
                trace_id, session_id, step_type, name, input_data, output_data,
                prompt_tokens, completion_tokens, latency_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,(
            trace_id,
            session_id,
            step_type,
            name,
            json.dumps(input_data) if isinstance(input_data, dict) else str(input_data) if input_data else None,
            json.dumps(output_data) if isinstance(output_data, dict) else str(output_data) if output_data else None,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            status
        ))
        conn.commit()

def db_get_trace(trace_id: str) -> list[dict]:
    """Retrieves all chronological steps recorded for a trace_id."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM traces WHERE trace_id = ? ORDER BY id ASC
        """,(trace_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

