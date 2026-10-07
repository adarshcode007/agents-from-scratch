### What the Messages look like after invoking tools

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful personal assistant."
    },
    {
        "role": "user",
        "content": "What time is it? Save a note saying I need to study."
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "arguments": "{}"
                }
            },
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "save_note",
                    "arguments": "{\"text\":\"I need to study\"}"
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "get_current_time",
        "content": "2026-10-06 01:04:52"
    },
    {
        "role": "tool",
        "tool_call_id": "call_456",
        "name": "save_note",
        "content": "Note saved: I need to study"
    },
    {
        "role": "assistant",
        "content": "The current date and time is 2026-10-06 01:04:52. This note has been saved.",
        "tool_calls": None
    }
]
```

### How we manage the conversation history without overblowing token usage?

we cap short-term history to the last 10–20 messages for three critical reasons:

- Cost: Input tokens are billed on every single request. If you send 50,000 tokens on each turn, costs balloon rapidly.
- Speed (Latency): Processing 50,000 tokens of history on every keystroke adds noticeable delay.
- "Lost in the Middle": LLMs get distracted and make tool-calling errors when context is cluttered with irrelevant chatter from 3 weeks ago.
The Golden Architecture:

Short-Term Memory: Last 10–20 messages from SQLite (sliding window).

Long-Term Memory: Durable facts stored separately and injected into the system prompt.


### Here is how FastAPI's BackgroundTasks works:

1. The Lifecycle (When it runs):
- When you call background_tasks.add_task(compact_session_history, req.session_id), FastAPI schedules the task.
- FastAPI first sends the 200 OK response JSON to the user immediately.
- Only AFTER the user has received their answer on their screen does FastAPI run compact_session_history in the background.

2. The User Experience Difference:
- Without BackgroundTasks (Synchronous):
    - Main chat model runs: 0.8 seconds
    - Summarizer model runs: 1.5 seconds
    - 👉 Total user waiting time: 2.3 seconds (user stares at a loading spinner for 2.3s)

- With BackgroundTasks:
    - Main chat model runs: 0.8 seconds
    - 👉 User gets the answer in 0.8s! 🚀
    - The summarizer runs silently in the background while the user is already reading their response.    



### reflect and critique with json output

sample input:
```json
{
    "task": "Write an apology email to enterprise clients about a 30-minute server downtime.",
    "criteria": [
        "Must offer a 15% discount on next month invoice",
        "Must explain that a database failover caused the issue",
        "Must be under 75 words in total length"
    ]
}
```

sample output:
```json
{
    "task": "Write an apology email to enterprise clients about a 30-minute server downtime.",
    "initial_draft": "Subject: Apology for Recent Downtime\n\nDear [Client],\n\nWe sincerely apologize for the 30‑minute outage on [date]. The interruption was caused by an unexpected database failover that disrupted services. We’ve resolved the issue and enhanced monitoring to prevent recurrence. As a token of our regret, we’ll apply a 15% discount to your next month’s invoice. Thank you for your understanding and continued partnership.\n\nBest regards,  \n[Name]  \n[Company]",
    "critique": {
        "passed": true,
        "critique": "The draft meets all required criteria: it offers a 15% discount, explains a database failover as the cause, and contains 61 words, which is under the 75‑word limit.",
        "issues_found": []
    },
    "final_output": "Subject: Apology for Recent Downtime\n\nDear [Client],\n\nWe sincerely apologize for the 30‑minute outage on [date]. The interruption was caused by an unexpected database failover that disrupted services. We’ve resolved the issue and enhanced monitoring to prevent recurrence. As a token of our regret, we’ll apply a 15% discount to your next month’s invoice. Thank you for your understanding and continued partnership.\n\nBest regards,  \n[Name]  \n[Company]",
    "refined": false
}
```