# RESEARCH_AND_ANALYSIS.md

**Status:** Initial plan only (not implemented yet). This outlines how I intend to meet the assignment requirements.

---

## Senior AI Engineer Task QSL-OC

### 1) System Architecture Overview

**Goal:** Build a document-aware AI chatbot with memory, document retrieval, and real-time streaming on Django + PostgreSQL + LangGraph.

**Models:** `ChatSession`, `ChatMessage`, `Document`, `DocumentChunk`.

**Endpoints:**
- `POST /api/chat/send/` → send message → synchronous AI reply (with LangGraph)
- `POST /api/chat/stream/` → send message → streamed AI reply (SSE)
- `POST /api/documents/upload/` → upload .txt for knowledge retrieval
- `GET /api/sessions/<id>/messages/` → fetch previous chat history

**Core requirements:** 
- Persist all sessions/messages
- Include prior history as context
- Orchestrate prompts/responses with LangGraph
- Use any LLM API

---

### 2) Technology Justifications

**Django + PostgreSQL:** Mandated backend stack and suitable for rapid REST + persistence.

**LangGraph:** Required to orchestrate LLM prompts/responses in a clear control flow.

**LLM provider:** Open choice (OpenAI, Claude, Groq, Gemini); I'll start with one provider and keep an adapter to swap if needed.

---

### 3) Document Retrieval Plan

**Input:** Accept `.txt` uploads.

**Processing:** Split into chunks (semantic or fixed-size).

**Search:** Compute embeddings (e.g., SentenceTransformers) and perform semantic search.

**Use in prompt:** Retrieve top relevant chunks and inject them into the LLM prompt for grounded answers.

---

### 4) Chat Memory Design

**Short-term context:** For each request, include a window of recent messages from the same session so the next AI response has history.

**Long-term summary (planned):** Maintain a rolling session summary stored on `ChatSession` to keep prompts compact while preserving context.

---

### 5) Streaming Implementation

**Requirement:** `/api/chat/stream/` returns token-by-token streamed responses (SSE).

**Plan:** Implement `POST /api/chat/stream/` as `text/event-stream` (SSE). The view will map provider deltas to JSON events: `data: {"type": "delta", "content": "..."}` and emit a final `data: {"type": "done", "message_id": "..."}`. Handle disconnects/errors gracefully and persist the final (or partial) assistant message.

---

### 6) Scalability & Extensibility

- Keep API stateless; rely on PostgreSQL for persistence.
- Add simple indices on foreign keys (`session_id`) and created-at fields for efficient history queries.
- Extensible later to additional text sources while keeping the same API surface (the spec only requires `.txt`).

---

### 7) Testing & Validation

**Unit:** 
- Chunking logic
- Embeddings adapter
- LangGraph node transitions (history → retrieval → synthesize)

**Integration:** 
- Upload `.txt` → chunk + embed → retrieve → inject chunks into prompt
- Verify history is included

**Streaming checks:** 
- Ensure `/api/chat/` emits multiple token events and completes cleanly
- Verify partial persistence on disconnect/error
- (Meets streaming behavior and endpoint expectations)

---

### 8) Deliverables Cross-Check

✅ Source code with migrations  
✅ This `RESEARCH_AND_ANALYSIS.md`  
✅ Working chat/document/history endpoints  
✅ README with setup and screenshots  
✅ `.env.example`  
✅ Pushed to a public GitLab per submission guidelines

---

## Implementation Phases

### Phase 0: Project Setup
- Django 5 skeleton
- Chat app structure
- PostgreSQL configuration

### Phase 1: Models & Persistence
- `ChatSession`, `ChatMessage` models
- Database migrations
- Admin interface

### Phase 2: Document Processing
- `Document`, `DocumentChunk` models
- Text chunking (fixed 500 char + 50 overlap)
- Embedding generation (SBERT)

### Phase 3: Retrieval System
- Cosine similarity search
- MMR (Maximal Marginal Relevance) for diverse results
- API endpoint for document upload

### Phase 4: Chat Core (Non-streaming)
- Synchronous chat endpoint
- OpenAI LLM integration
- RAG (Retrieval-Augmented Generation)
- History inclusion in prompts

### Phase 5: LangGraph Orchestration
- State management
- Node-based workflow (load_history → decide_retrieve → retrieve → synthesize → summarize)
- Conditional retrieval logic
- Graph compilation

### Phase 6: Streaming (SSE)
- Server-Sent Events implementation
- Token-by-token streaming
- Browser demo UI
- Error handling & partial persistence

### Phase 7: Memory Management
- Token-bounded history (max 3000 tokens, min 6 turns)
- Rolling session summary (every 5 assistant turns)
- Long-term memory via summarization

---

## Key Design Decisions

### 1. Embedding Strategy
- **Choice:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Rationale:** Fast, open-source, no API costs, good quality for semantic search

### 2. Chunking Strategy
- **Choice:** Fixed-size (500 chars) with overlap (50 chars)
- **Rationale:** Simple, predictable, handles arbitrary text well

### 3. Retrieval Algorithm
- **Choice:** MMR (Maximal Marginal Relevance)
- **Rationale:** Balances relevance with diversity, avoids redundant chunks

### 4. LLM Provider
- **Choice:** OpenAI (gpt-4o-mini)
- **Rationale:** Excellent quality, streaming support, cost-effective

### 5. Memory Management
- **Choice:** Token-bounded history + rolling summaries
- **Rationale:** Stays within LLM context limits while preserving long conversations

### 6. Orchestration
- **Choice:** LangGraph
- **Rationale:** Clear state management, testable nodes, conditional logic

---

## Expected Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| **Large documents** | Chunk into 500-char pieces with overlap |
| **Context window limits** | Token-bounded history + summarization |
| **Streaming complexity** | Manual node orchestration for streaming path |
| **Memory overhead** | Store embeddings in JSONField, use PostgreSQL for scaling |
| **LLM costs** | Use gpt-4o-mini, implement caching (future) |
| **Testing streaming** | Mock LLM responses, simulate disconnects |

---

## Success Criteria

✅ **Functional Requirements:**
- Chat endpoint with streaming responses
- Document upload with automatic chunking/embedding
- Conversation history in context
- RAG for grounded answers
- Session persistence across requests

✅ **Technical Requirements:**
- LangGraph orchestration
- PostgreSQL persistence
- Comprehensive tests (unit + integration)
- Error handling
- CORS support for browser demo

✅ **Deployment Requirements:**
- Environment variable configuration
- Production-ready settings
- Clear documentation
- Public GitLab repository

---

*This document serves as the initial research and analysis for the QSL Senior AI Engineer assignment. Implementation will follow these guidelines while adapting to discoveries during development.*

