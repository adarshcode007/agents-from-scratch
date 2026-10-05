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