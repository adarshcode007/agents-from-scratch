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

### Q11: Why run Rolling Summarization only when a threshold is reached, instead of on every turn?
**A:** 
1. **Speed & Latency**: Summarizing on every user message doubles the API calls and adds latency to every response. Running it only when history exceeds a threshold (e.g., every 10+ turns) keeps normal requests fast.
2. **Cost Efficiency**: Reduces token burn by compacting batches of messages at once rather than re-summarizing after every single sentence.
3. **Separation of Concerns**: Keeps the main agent loop focused purely on user interaction and tool execution, while a dedicated prompt handles concise distillation.

---

### Q12: How does Rolling Summarization scale when a conversation reaches 200+ messages?
**A:** 
- Instead of reprocessing all 200 messages, it takes the **existing summary** (which already condensed messages 1–190) plus the **latest chunk of messages (191–200)** to generate the updated summary.
- This achieves **$O(1)$ constant prompt size** for compaction, meaning context compression costs stay flat whether the conversation has 50 turns or 5,000 turns.

---

### Q13: What is the fundamental difference between Keyword Search and Semantic Vector Search?
**A:** 
- **Keyword Search (`LIKE '%...%'` / BM25)**: Matches exact words or character substrings. It fails when different words share the same meaning (e.g., `"car"` vs. `"automobile"`).
- **Semantic Vector Search**: Uses an embedding model to convert text into mathematical coordinates (vectors). Texts with similar conceptual meanings have high **cosine similarity** (close distance in vector space), enabling meaning-based retrieval regardless of wording.

---

### Q14: What are Embeddings, what does Similarity Threshold mean, and how does FastEmbed compare to OpenAI?
**A:** 
- **Embeddings**: High-dimensional numerical vectors representing text meaning, where conceptual similarity corresponds to geometric proximity (measured via Cosine Similarity from `-1.0` to `1.0`).
- **Threshold (e.g. 0.3)**: A noise filter. Prevents forcing irrelevant memories into the context window when the user's message has no semantic connection to stored facts.
- **FastEmbed vs OpenAI**: FastEmbed runs locally on CPU via ONNX (~2-5ms latency, $0 cost, completely private, 384-dim). OpenAI embeddings require an external API call (~200ms latency, token cost, 1536-dim).

---

### Q15: Why use Structured Outputs (Pydantic / JSON Mode) instead of prompt-based instructions like "Return JSON"?
**A:** 
- Prompting *"Return JSON only"* is prone to failures: models often add markdown ticks (````json ... ````), omit required keys, or generate wrong data types.
- **Structured Outputs** enforce a mathematical grammar constraint during LLM token generation, guaranteeing 100% syntactically valid JSON that is automatically validated against a strict Pydantic model before downstream execution.

---

### Q16: Why does the Plan-and-Solve / Scratchpad pattern outperform pure ReAct on complex tasks?
**A:** 
- Pure ReAct is greedy and easily forgets multi-step goals, executes steps out of order, or gets trapped in redundant tool loops.
- **Plan-and-Solve** creates an explicit checklist in context (*Working Memory*), enabling the agent to track progress (`[x]` vs `[ ]`), resolve step dependencies, and stay focused across 5+ iterations.

---

### Q17: Why is a separate Critic (Reflection loop) superior to single-pass generation?
**A:** 
- **Single-Pass Blind Spot**: LLMs generate text token-by-token and cannot backtrack to fix mistakes made earlier in the paragraph.
- **Evaluator-Optimizer Pattern**: Isolates creation from quality assurance. The Generator produces a candidate draft, and the Critic systematically checks the completed text against constraints (tone, length, required facts), allowing a Refiner step to correct flaws before the user sees the output.

---

### Q18: What if the Refiner also makes a mistake during reflection?
**A:** 
1. **Iterative Refinement**: Wrap the Critic $\rightarrow$ Refiner cycle in a bounded `while` loop (e.g. `max_retries = 3`), giving the refiner up to 3 attempts to satisfy the critic.
2. **Deterministic Code Gates**: Augment LLM critique with deterministic Python code assertions (word count checks, regex rules, unit tests) for 100% mathematical guarantees.
3. **Human Escalation (HITL)**: If multiple refinement rounds fail, flag the item for human review rather than outputting unchecked failures.

---











