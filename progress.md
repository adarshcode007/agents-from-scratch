# Applied AI / Agentic AI Engineer Career Roadmap & Progress Tracker

This document maps out the essential competencies, architectural patterns, and production-grade skills required to get hired as an **Applied AI / Agentic AI Engineer**.

---

## 📍 Current Status: `Phase 2 (Planning, Structured Outputs & Reasoning)` — Ready to Start 🚀

```
[✅ Phase 0] ──▶ [✅ Phase 1] ──▶ [🟡 Phase 2] ──▶ [⏳ Phase 3] ──▶ [⏳ Phase 4] ──▶ [⏳ Phase 5] ──▶ [💼 Portfolio]
Core Engine      Memory/Context    Planning/Evals    Multi-Agent       Production Ops    Advanced RAG      Capstones
```

---

## 🗺️ Detailed Syllabus & Roadmap

### Phase 0: The Core Agent Engine & Function Calling (COMPLETED ✅)
*Mastering agent loops and tool protocols from first principles without frameworks.*
- [x] **Handcrafted ReAct Loop**: Built the core `while` loop (LLM call $\rightarrow$ parse tool calls $\rightarrow$ execute Python tool $\rightarrow$ feed tool output $\rightarrow$ repeat until text).
- [x] **JSON Schema Tool Declarations**: Declaring typed schemas (Pydantic / OpenAI format) with descriptions and parameter types.
- [x] **Defensive Error Handling**: Catching `JSONDecodeError`, `TypeError`, and runtime exceptions to let the LLM self-correct rather than crash.
- [x] **Safety Guards**: Loop iteration limits (`max_iterations`) to prevent infinite recursion and token drain.
- [x] **Tool Side Effects**: Modifying runtime state and databases (CRUD operations on tasks and notes).

---

### Phase 1: Memory Architectures & Context Engineering (COMPLETED ✅)
*Giving agents short-term conversational context and long-term persistent recall.*
- [x] **Short-Term Session History**: Storing raw multi-turn conversation logs in SQLite and querying with sliding windows (`LIMIT 6`).
- [x] **Long-Term Memory Injection**: The `remember(fact)` tool saving durable user traits and injecting them into the system prompt across sessions.
- [x] **Context Window Compaction / Rolling Summarization**: Triggering background LLM summarization when chat history exceeds token limits so the agent never forgets older topics.
- [x] **Semantic Memory Search (Vector Embeddings)**: Storing memories with embeddings (`FastEmbed` ONNX) and cosine similarity to retrieve the top-$K$ most relevant facts dynamically.

---


### Phase 2: Structured Outputs, Planning & Task Decomposition (IN PROGRESS 🟡 - YOU ARE HERE)
*Moving from reactive tool-calling to proactive multi-step problem solving.*
- [x] **Structured Outputs / JSON Mode**: Enforcing strict Pydantic return schemas from the LLM without freeform formatting errors.
- [x] **Plan-and-Solve / Scratchpad Pattern**: Prompting the agent to write a structured execution plan before calling tools, updating step status along the way.
- [x] **Reflection & Self-Verification**: The agent evaluates its own output against constraints before sending the final answer to the user.
- [ ] **Human-in-the-Loop (HITL) Approval Gates**: Requiring explicit user confirmation before executing irreversible or high-risk side effects (e.g. database deletes, payments, emails).

---




### Phase 3: Multi-Agent Systems & Orchestration (UPCOMING ⏳)
*Building networks of specialized agents that collaborate on complex tasks.*
- [ ] **Orchestrator–Worker Pattern**: A supervisor agent delegates sub-tasks to dedicated specialist agents (e.g. Research Agent, Coder Agent, Reviewer Agent).
- [ ] **Agent Handoffs & State Graphs**: Routing conversation control between agents using state machines (similar to LangGraph / Swarm architectures).
- [ ] **Subagent Isolated Contexts**: Spawning focused subagents with restricted tools and summarized report handoffs to prevent context clutter.

---

### Phase 4: Production Engineering, Observability & Evals (UPCOMING ⏳)
*What differentiates an amateur hobbyist from a senior Applied AI Engineer.*
- [ ] **Tracing & Observability**: Logging step-by-step token usage, tool latency, and agent thoughts to a structured telemetry table or dashboard (e.g. Langfuse / OpenTelemetry).
- [ ] **Streaming Responses (SSE / WebSockets)**: Streaming tokens and real-time tool execution status to the frontend.
- [ ] **Automated Eval Suites (CI/CD for Agents)**: Writing 15–20 deterministic benchmark tests (e.g., verifying that prompt $X$ triggered tool $Y$ with parameter $Z$ and left the database in state $S$).
- [ ] **LLM-as-a-Judge Evaluation**: Programmatically grading response quality, accuracy, and tone against gold-standard rubrics.

---

### Phase 5: Advanced RAG (Retrieval-Augmented Generation) (UPCOMING ⏳)
*Connecting agents to large external unstructured knowledge bases.*
- [ ] **Document Chunking & Embedding Strategies**: Parent-document retrieval, recursive text splitting, and metadata enrichment.
- [ ] **Hybrid Search (Dense + Sparse)**: Combining BM25 keyword search with vector semantic search using Reciprocal Rank Fusion (RRF).
- [ ] **Re-ranking & Query Transformation**: Rewriting user queries and applying Cross-Encoder re-rankers to improve top-$K$ precision.

---

### Phase 6: Capstone Projects for Job Applications (UPCOMING ⏳)
*Production portfolio projects to present in interviews.*
1. **Autonomous Personal Assistant with Telegram Bot** (The complete evolution of this project).
2. **Autonomous Multi-Agent Research & Report Generator** (Web search $\rightarrow$ fact extraction $\rightarrow$ cross-checking $\rightarrow$ formatted markdown PDF generation).
3. **Customer Support / Codebase Copilot with RAG & Guardrails** (Enterprise knowledge search, permission checks, and transactional tool execution).

---
