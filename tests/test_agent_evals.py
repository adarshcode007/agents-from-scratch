import pytest
from uuid import uuid4
import db


# SUITE 1: Tool Selection & State Mutation (Deterministic Evals)

def test_eval_save_note_tool(client):
    """Eval: Prompting the agent to save a note must invoke save_note and persist to DB."""
    session_id = f"test_{uuid4().hex[:8]}"
    secret_note = "Quantum computing uses qubits instead of classical bits."

    resp = client.post("/chat", json={
        "session_id": session_id,
        "message": f"Please save this note for me: '{secret_note}'"
    })

    assert resp.status_code == 200
    # Invariant: The note must physically exist in SQLite
    notes_in_db = db.db_search_notes("Quantum computing")
    assert len(notes_in_db) >= 1
    assert any(secret_note in n for n in notes_in_db)

def test_eval_add_task_tool(client):
    """Eval: Prompting the agent to add a task must create a pending task in DB."""
    session_id = f"test_{uuid4().hex[:8]}"
    resp = client.post("/chat", json={
        "session_id": session_id,
        "message": "Add a new task: Prepare the Q3 production deployment by tomorrow"
    })

    assert resp.status_code == 200
    # Invariant: The task must exist in pending tasks
    pending = db.db_list_tasks("pending")
    assert any("Q3" in t["title"] or "deployment" in t["title"].lower() for t in pending)


# SUITE 2: Negative Constraints & Guardrails (No Tools Triggered)

def test_eval_negative_constraint_general_knowledge(client):
    """Eval: General knowledge prompts must answer directly without touching tools/notes."""
    session_id = f"test_{uuid4().hex[:8]}"
    notes_before = len(db.db_search_notes(""))
    tasks_before = len(db.db_list_tasks("pending"))

    resp = client.post("/chat", json={
        "session_id": session_id,
        "message": "What is the capital of France? Answer in one sentence."
    })

    assert resp.status_code == 200
    data = resp.json()
    assert "Paris" in data["response"]

    # Invariant: No Tools should have been triggered (no new notes or tasks)
    assert len(db.db_search_notes("")) == notes_before
    assert len(db.db_list_tasks("pending")) == tasks_before

# SUITE 3: Multi-Agent Swarm Handoff Routing

def test_eval_swarm_technical_routing(client):
    """Eval: Technical server/database issues must route to SupportAgent."""
    resp = client.post("/multi-agent/swarm", json={
        "message": "Our PostgreSQL database server is crashing with connection timeouts."
    })
    assert resp.status_code == 200
    data = resp.json()
    # Invariant 1: Final agent or handoff history must involve SupportAgent
    assert data["final_agent"] == "SupportAgent" or any(
        h["to_agent"] == "SupportAgent" for h in data["handoff_history"]
    )

def test_eval_swarm_sales_routing(client):
    """Eval: Pure sales inquiries must route directly to SalesAgent."""
    resp = client.post("/multi-agent/swarm", json={
        "message": "I want to schedule an enterprise demo and discuss pricing plans."
    })
    assert resp.status_code == 200
    data = resp.json()
    # Invariant: Handoff history must route to SalesAgent
    assert data["final_agent"] == "SalesAgent" or any(
        h["to_agent"] == "SalesAgent" for h in data["handoff_history"]
    )



# SUITE 4: Observability & Telemetry Invariants


def test_eval_trace_observability_logged(client):
    """Eval: Every swarm execution must produce a trace with token metrics and latency."""
    resp = client.post("/multi-agent/swarm", json={
        "message": "I need help with server logs."
    })
    assert resp.status_code == 200
    trace_id = resp.json()["trace_id"]
    assert trace_id.startswith("tr_")
    # Invariant: The trace details must be fetchable and contain non-zero metrics
    trace_resp = client.get(f"/traces/{trace_id}")
    assert trace_resp.status_code == 200
    metrics = trace_resp.json()
    assert metrics["total_steps"] >= 1
    assert metrics["total_prompt_tokens"] > 0
    assert metrics["total_latency_ms"] > 0
    assert len(metrics["steps"]) >= 1

