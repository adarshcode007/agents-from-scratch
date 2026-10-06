# Q&A / Concept Log

### Q4: Why return error strings from tools instead of raising Python exceptions?
**A:** 
- If a tool raises an exception (`raise Exception`), the server crashes with a `500` error and terminates the conversation.
- If a tool returns a descriptive string (e.g. `"Error: Note 99 not found"`), the LLM reads that output in the next loop iteration, understands what went wrong, and can **self-correct** or gracefully explain the issue to the user.

---

### Q5: Why have a separate `memories` table when we already store chat history in `messages`?
**A:** 
- **Chat history (`messages`)** uses a sliding window (e.g. last 10 turns) to prevent latency and token cost blowups, meaning facts mentioned 20+ turns ago fall out of context.
- **Long-term memory (`memories`)** stores extracted, durable facts (e.g. preferences, allergies) that are injected into the system prompt on *every* request, so they persist permanently across sessions regardless of conversation length.

---

### Q8: Why are Side-Effect Tools higher risk than Read-Only Tools?
**A:** 
- **Read-Only tools** (like `get_current_time`, `search_notes`) only retrieve data without changing state; calling them repeatedly or incorrectly is mostly harmless.
- **Side-Effect tools** (like `save_note`, `delete_task`, `send_email`) mutate external state (writing to DB, modifying data, calling external APIs). If tool descriptions or schemas are ambiguous, the LLM might trigger unintended writes, overwrite data, or corrupt state.

---

### Q9: How should an agent handle malformed JSON or invalid tool arguments from the LLM?
**A:** 
- Wrap `json.loads()` and tool execution (`tool_func(**tool_args)`) in `try...except` blocks.
- Instead of letting `JSONDecodeError` or `TypeError` crash the server, catch them and return the error string back to the LLM as the tool result.
- The LLM receives the error message on its next loop iteration and can automatically self-correct its arguments.

---

### Q10: Why is `due_date` stored as a string (`TEXT`) in SQLite?
**A:** 
- SQLite has no native `DATETIME` storage class (only `NULL`, `INTEGER`, `REAL`, `TEXT`, `BLOB`).
- Standard ISO-8601 strings (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`) are:
  1. **LLM Native**: Models easily output and parse ISO date strings.
  2. **Sortable**: Alphabetical sorting in SQL matches chronological order (`ORDER BY due_date`).
  3. **Compatible**: SQLite's built-in `datetime()` and `strftime()` functions operate directly on ISO strings.

---



