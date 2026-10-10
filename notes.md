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


### POST /multi-agent/orchestrate

sample input:
```json
{
  "goal": "Check our notes and memory facts about Python study goals or server downtime. Then create an actionable follow-up task for tomorrow in my tasks database. Finally, give me an executive briefing of what you found and scheduled."
}
```

sample output:
```json
{
    "goal": "Check our notes and memory facts about Python study goals or server downtime. Then create an actionable follow-up task for tomorrow in my tasks database. Finally, give me an executive briefing of what you found and scheduled.",
    "subagent_reports": [
        {
            "agent_name": "ResearchWorker",
            "task_input": "Retrieve notes and memory facts about Python study goals and server downtime.",
            "report_output": "**Research Briefing – Python Study Goals & Server Downtime**\n\n- **Python Study Goals**  \n  - *Notes:* None found in the user’s notes.  \n  - *Memory facts:*  \n    - The user prefers TypeScript over Python for web projects.  \n    - No specific Python learning objectives or milestones were recorded.\n\n- **Server Downtime**  \n  - *Notes:* None found in the user’s notes.  \n  - *Memory facts:*  \n    - The same two facts appear (TypeScript preference and table‑tennis interest); no direct mention of server downtime issues.\n\n**Summary:**  \nThe user’s stored data does not contain explicit notes about Python study plans or server‑downtime incidents. The only relevant memory facts highlight a preference for TypeScript over Python in web development and a personal interest in table tennis. No concrete details on study goals or downtime events are available. If more detailed information is needed, please provide additional context or specific topics.",
            "iterations_used": 5
        },
        {
            "agent_name": "TaskWorker",
            "task_input": "Create an actionable follow-up task for tomorrow: Review and document Python study goals and assess server uptime logs. Include steps: 1) Meet with team to outline Python objectives. 2) Pull server logs from last week. 3) Identify any downtime incidents. Deadline: Tomorrow, 10:00 AM. Label: \"Python & Server Review\".",
            "report_output": "**Task Summary**\n\n| Field | Value |\n|-------|-------|\n| **Task ID** | *generated by the system* |\n| **Title** | Python & Server Review |\n| **Due Date** | 2026-10-09 |\n| **Description** | Actionable follow-up task: Review and document Python study goals and assess server uptime logs.<br><br>**Steps:**<br>1) Meet with team to outline Python objectives.<br>2) Pull server logs from last week.<br>3) Identify any downtime incidents.<br><br>Deadline: Tomorrow, 10:00 AM. |\n| **Status** | Pending |\n\n*The task has been added to your task list and is due tomorrow at 10:00 AM.*",
            "iterations_used": 3
        }
    ],
    "final_synthesis": "**Executive Briefing – Findings & Next Steps**\n\n| Area | Findings | Action Taken |\n|------|----------|--------------|\n| **Python Study Goals** | No specific notes or memories were found in your records. The only related memory fact is that you prefer TypeScript over Python for web projects, with no defined learning objectives. | A follow‑up task has been created to clarify and document your Python study goals. |\n| **Server Downtime** | No notes or memories regarding server downtime incidents were retrieved. The only associated memory fact is a general interest in table tennis, unrelated to server operations. | The same follow‑up task will include pulling server logs and checking for any recent downtime incidents. |\n\n---\n\n### Actionable Task Created\n\n**Task ID:** *(system generated)*  \n**Title:** **Python & Server Review**  \n**Due Date:** **2026‑10‑09 (Tomorrow)**  \n**Deadline:** **10:00 AM**  \n**Description & Steps:**\n\n1. **Meet with the team** to outline and define concrete Python study objectives.  \n2. **Pull server logs** from the last week and review uptime/downtime metrics.  \n3. **Identify any downtime incidents** and document root causes or mitigation steps.  \n\nThe task is now in your task database and will appear in your schedule for tomorrow at the specified time. If you need more detail or have additional context, just let me know!"
}
```



### POST /multi-agent/swarm

Sample Input:
```json
{
  "message": "I have a technical question about our database server downtime. Check my saved notes to see what was recorded about it, and then hand me off to Sales to schedule a follow-up meeting for tomorrow to review the system."
}

```

Sample Output:
```json
{
    "trace_id": "tr_f2862g16",     // added later
    "final_agent": "SalesAgent",
    "handoff_history": [
        {
            "from_agent": "TriageAgent",
            "to_agent": "SupportAgent",
            "reason": "Transferred via tool 'transfer_to_support'"
        },
        {
            "from_agent": "SupportAgent",
            "to_agent": "SalesAgent",
            "reason": "Transferred via tool 'transfer_to_sales'"
        }
    ],
    "final_response": "I’ve added a task for a follow-up meeting to review the database server downtime tomorrow (October 10, 2026). I’ll hand you over to Sales to coordinate the details. Have a great day!"
}
```


### GET /traces/{trace_id}

Sample Output:
```json
{
  "trace_id": "tr_f2862f15",
  "total_steps": 7,
  "total_prompt_tokens": 1549,
  "total_completion_tokens": 221,
  "total_latency_ms": 1891.22,
  "steps": [
    {
      "id": 1,
      "trace_id": "tr_f2862f15",
      "session_id": null,
      "step_type": "llm_call",
      "name": "TriageAgent (LLM)",
      "input_data": "{\"message_count\": 2}",
      "output_data": "{\"text\": null, \"tool_calls_count\": 1}",
      "prompt_tokens": 230,
      "completion_tokens": 39,
      "latency_ms": 421.8933999654837,
      "status": "success",
      "created_at": "2026-10-10 11:10:02"
    },
    {
      "id": 2,
      "trace_id": "tr_f2862f15",
      "session_id": null,
      "step_type": "handoff",
      "name": "Handoff: TriageAgent -> SalesAgent",
      "input_data": "{\"from\": \"TriageAgent\", \"to\": \"SalesAgent\"}",
      "output_data": null,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "latency_ms": 0.0,
      "status": "success",
      "created_at": "2026-10-10 11:10:02"
    },
    {
      "id": 3,
      "trace_id": "tr_f2862f15",
      "session_id": null,
      "step_type": "llm_call",
      "name": "SalesAgent (LLM)",
      "input_data": "{\"message_count\": 4}",
      "output_data": "{\"text\": null, \"tool_calls_count\": 1}",
      "prompt_tokens": 340,
      "completion_tokens": 98,
      "latency_ms": 548.2515000039712,
      "status": "success",
      "created_at": "2026-10-10 11:10:02"
    },
    {
      "id": 4,
      "trace_id": "tr_f2862f15",
      "session_id": null,
      "step_type": "tool_execution",
      "name": "get_current_time",
      "input_data": "{}",
      "output_data": "2026-10-10 16:40:02",
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "latency_ms": 0.04020001506432891,
      "status": "success",
      "created_at": "2026-10-10 11:10:02"
    },
    {
      "id": 5,
      "trace_id": "tr_f2862f15",
      "session_id": null,
      "step_type": "llm_call",
      "name": "SalesAgent (LLM)",
      "input_data": "{\"message_count\": 6}",
      "output_data": "{\"text\": null, \"tool_calls_count\": 1}",
      "prompt_tokens": 457,
      "completion_tokens": 50,
      "latency_ms": 273.2374999905005,
      "status": "success",
      "created_at": "2026-10-10 11:10:02"
    },
    {
      "id": 6,
      "trace_id": "tr_f2862f15",
      "session_id": null,
      "step_type": "tool_execution",
      "name": "add_task",
      "input_data": "{\"description\": \"Appointment at dentist office\", \"due_date\": \"2026-10-16T15:00:00\", \"title\": \"Dentist Appointment\"}",
      "output_data": "Task added: Dentist Appointment",
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "latency_ms": 3.2632999937050045,
      "status": "success",
      "created_at": "2026-10-10 11:10:02"
    },
    {
      "id": 7,
      "trace_id": "tr_f2862f15",
      "session_id": null,
      "step_type": "llm_call",
      "name": "SalesAgent (LLM)",
      "input_data": "{\"message_count\": 8}",
      "output_data": "{\"text\": \"Your dentist appointment has been scheduled for next Tuesday, October\\u202f16th at 3\\u202fp.m. Let me know if you need anything else!\", \"tool_calls_count\": 0}",
      "prompt_tokens": 522,
      "completion_tokens": 34,
      "latency_ms": 644.5295999874361,
      "status": "success",
      "created_at": "2026-10-10 11:10:03"
    }
  ]
}

```


### POST /multi-agent/stream

Sample input:
```json
{
  "message": "I need to check our notes for any database downtime issues, and then schedule a follow-up meeting with sales for tomorrow."
}
```

Sample Output: (check in postman)

