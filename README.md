# ChatServer — RAG-Powered Chatbot Backend

Django 5 backend with document ingestion, semantic search (with MMR), LangGraph orchestration, and streaming chat capabilities.

🚀 **Live Demo**: [https://191.101.81.150](https://191.101.81.150) ← **Use this for public access**  
💻 **Local Development**: [http://localhost:8000/demo/](http://localhost:8000/demo/) (when running locally)

## Features

- 📄 **Document Management**: Upload and process text documents (500KB limit, .txt only)
- 🔍 **Semantic Search**: SBERT embeddings with cosine similarity
- 🎯 **MMR (Maximal Marginal Relevance)**: Diverse, relevant retrieval results
- 💬 **Session-Based Chat**: Independent sessions with document binding
- 🧠 **Smart Memory**: Token-bounded history (3000 tokens) + rolling summaries (every 5 turns)
- 🔄 **LangGraph Orchestration**: Multi-node workflow with intelligent retrieval decisions
- ⚡ **Real-time Streaming**: Server-Sent Events (SSE) for token-by-token responses
- 🗄️ **PostgreSQL Database**: Production-ready database with session isolation
- 🎯 **Intelligent Retrieval**: Automatic RAG triggering based on message content

---

## Quick Start

### 1. Prerequisites

**PostgreSQL Database:**
```bash
# Option 1: Using Docker (recommended for development)
docker run -d \
  --name qtec-postgres \
  -e POSTGRES_DB=qtec_chatbot \
  -e POSTGRES_USER=qtec_user \
  -e POSTGRES_PASSWORD=qtec_password \
  -p 5433:5432 \
  postgres:15

# Option 2: Install PostgreSQL locally
# Follow instructions at https://www.postgresql.org/download/
```

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda activate legal_assistant
pip install -r requirements.txt

# Configure environment variables
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run migrations
python manage.py migrate

# Create superuser (for admin access)
python manage.py createsuperuser
```

### 3. Run Server

```bash
python manage.py runserver
```

### 4. Health Check

```bash
curl http://localhost:8000/health/
# {"status": "ok"}
```

---

## API Endpoints

### Health

#### GET `/health/`
Health check endpoint.

```bash
curl http://localhost:8000/health/
```

**Response:**
```json
{
  "status": "ok"
}
```

---

### Documents

#### POST `/api/documents/upload/`
Upload and process a document. Documents are bound to chat sessions for isolation.

```bash
curl -X POST http://localhost:8000/api/documents/upload/ \
  -F "file=@document.txt" \
  -F "auto_process=true" \
  -F "session_id=session-uuid"
```

**Parameters:**
- `file` (required): Text file (.txt only, max 500KB)
- `auto_process` (optional, default: true): Automatically chunk and embed
- `session_id` (optional): Bind document to specific session

**Response:**
```json
{
  "id": "uuid-here",
  "filename": "document.txt",
  "file_size": 1024,
  "chunk_count": 5,
  "auto_processed": true,
  "session_id": "session-uuid",
  "created_at": "2025-10-08T12:00:00Z"
}
```

**Note:** Uploading a new document to an existing session will delete previous documents and chat history for that session.

#### GET `/api/documents/`
List all documents.

```bash
curl http://localhost:8000/api/documents/
```

**Response:**
```json
[
  {
    "id": "uuid",
    "filename": "document.txt",
    "raw_text": "Full document text...",
    "chunk_count": 5,
    "created_at": "2025-10-08T12:00:00Z"
  }
]
```

#### GET `/api/documents/{id}/`
Get a specific document.

```bash
curl http://localhost:8000/api/documents/{document-id}/
```

#### GET `/api/documents/{id}/chunks/`
Get all chunks for a document.

```bash
curl http://localhost:8000/api/documents/{document-id}/chunks/
```

**Response:**
```json
[
  {
    "id": "chunk-uuid",
    "chunk_index": 0,
    "text": "This is chunk text...",
    "has_embedding": true,
    "created_at": "2025-10-08T12:00:00Z"
  }
]
```

---

### Retrieval (Semantic Search)

#### POST `/api/retrieve/`
Search for relevant document chunks using semantic similarity.

**With MMR (recommended for diversity):**
```bash
curl -X POST http://localhost:8000/api/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 5,
    "use_mmr": true,
    "lambda_param": 0.5
  }'
```

**Without MMR (pure relevance):**
```bash
curl -X POST http://localhost:8000/api/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 3,
    "use_mmr": false
  }'
```

**Filter by session:**
```bash
curl -X POST http://localhost:8000/api/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search text",
    "top_k": 5,
    "session_id": "session-uuid"
  }'
```

**Parameters:**
- `query` (required): Search query text
- `top_k` (optional, default: 3): Number of results to return
- `use_mmr` (optional, default: true): Enable MMR for diversity
- `lambda_param` (optional, default: 0.5): MMR trade-off (0=diversity, 1=relevance)
- `session_id` (optional): Filter by specific session (retrieves only session's documents)

**Response:**
```json
{
  "query": "machine learning",
  "top_k": 3,
  "use_mmr": true,
  "lambda_param": 0.5,
  "results": [
    {
      "score": 0.8532,
      "chunk_id": "chunk-uuid",
      "text": "Machine learning is a subset of AI...",
      "document_id": "doc-uuid",
      "document_filename": "ml-basics.txt",
      "chunk_index": 2
    }
  ]
}
```

---

### Chat Sessions

#### GET `/api/sessions/`
List all chat sessions.

```bash
curl http://localhost:8000/api/sessions/
```

#### POST `/api/sessions/`
Create a new chat session.

```bash
curl -X POST http://localhost:8000/api/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"title": "New Conversation"}'
```

#### GET `/api/sessions/{id}/`
Get a specific session with messages.

```bash
curl http://localhost:8000/api/sessions/{session-id}/
```

---

### Chat (AI Conversation)

#### POST `/api/chat/send/` (Synchronous)
Send a message and get AI response with RAG (Retrieval-Augmented Generation).

#### POST `/api/chat/stream/` (Streaming - SSE)
Send a message and receive AI response via Server-Sent Events (streaming). Retrieval is automatically triggered based on message content.

**Request:**
```bash
curl -X POST http://localhost:8000/api/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-uuid",
    "message": "What is machine learning?",
    "model": "gpt-4o-mini"
  }'
```

**Parameters:**
- `session_id` (required): Chat session UUID
- `message` (required): User's message/question
- `model` (optional, default: gpt-4o-mini): OpenAI model to use

**Retrieval Triggering:**
The system automatically determines when to use RAG based on:
- Question words: "what", "how", "why", "when", "where", "who", "explain", "tell", "describe", "give", "show", "list", "provide"
- Question marks: Any message ending with "?"
- Long messages: More than 10 words
- Document references: Messages containing "document", "file", "source", "according to", "based on"

**Response:**
```json
{
  "session_id": "session-uuid",
  "message_id": "message-uuid",
  "content": "Machine learning is a subset of artificial intelligence...",
  "retrieved_chunks": [
    {
      "text": "ML definition from docs...",
      "score": 0.89,
      "document": "ml_basics.txt",
      "chunk_id": "chunk-uuid"
    }
  ],
  "metadata": {
    "tokens_used": 450,
    "retrieval_count": 3,
    "context_messages": 2,
    "model": "gpt-4o-mini"
  }
}
```

**Error Responses:**
```json
{
  "error": "Session not found",
  "code": "SESSION_NOT_FOUND"
}

{
  "error": "LLM authentication failed. Please check API key configuration.",
  "code": "LLM_AUTH_ERROR"
}

{
  "error": "Rate limit exceeded. Please try again later.",
  "code": "RATE_LIMIT_EXCEEDED"
}
```

**Streaming Example (SSE):**

The `/api/chat/stream/` endpoint returns Server-Sent Events for real-time streaming:

```bash
curl -X POST http://localhost:8000/api/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-uuid",
    "message": "Explain quantum computing",
    "model": "gpt-4o-mini"
  }'
```

**SSE Response Format:**
```
data: {"type": "delta", "content": "Quantum"}
data: {"type": "delta", "content": " computing"}
data: {"type": "delta", "content": " is..."}
data: {"type": "done", "message_id": "msg-uuid", "chunks": 3}
```

Event types:
- `delta`: Text chunk from LLM (stream these to display)
- `done`: Streaming complete (message saved to DB)
- `error`: Error occurred during streaming

**Browser Demo:**  
Open `static/sse-demo.html` in a browser for a live streaming chat demo.

---

## Management Commands

### Ingest Documents
Process all documents (chunk + embed) in bulk:

```bash
python manage.py ingest_docs
```

This will:
1. Find all documents without embeddings
2. Chunk the text
3. Generate embeddings using SBERT
4. Store in database

---

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Django
SECRET_KEY=your-secret-key
DEBUG=1
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (optional - defaults to SQLite)
DB_NAME=qsl_chatbot
DB_USER=qsl_user
DB_PASSWORD=qsl_pass
DB_HOST=localhost
DB_PORT=5432

# OpenAI (REQUIRED for chat functionality)
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini

# Chat Configuration
CHAT_CONTEXT_MESSAGES=10
CHAT_MAX_TOKENS=2000
CHAT_TEMPERATURE=0.7

# RAG Configuration
RAG_ENABLED=true
RAG_TOP_K=3
RAG_USE_MMR=true

# Memory Configuration (Phase 7)
CHAT_MAX_TOKENS_CONTEXT=3000
CHAT_HISTORY_MIN_TURNS=6
SUMMARY_INTERVAL_TURNS=5
```

**Important:** The chat endpoint requires a valid OpenAI API key. Get one from [OpenAI Platform](https://platform.openai.com/api-keys).

**Memory Features:**
- **Token-bounded history**: Limits conversation context by token count (default: 3000 tokens)
- **Minimum turns**: Ensures at least N turns are kept regardless of token limit (default: 6)
- **Auto-summarization**: Creates long-term summaries every N assistant turns (default: 5)

### Database

**PostgreSQL (Production):**
- **Database**: `qtec_chatbot`
- **User**: `qtec_user`
- **Port**: `5433` (Docker) or `5432` (local)
- **Connection pooling**: Enabled with `CONN_MAX_AGE=60`
- **Driver**: `psycopg2-binary` (included in requirements)

**Configuration** (`chatserver/settings.py`):
```python
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "qtec_chatbot",
        "USER": "qtec_user",
        "PASSWORD": "qtec_password",
        "HOST": "localhost",
        "PORT": "5433",
        "CONN_MAX_AGE": 60,
    }
}
```

**Benefits:**
- ✅ Better concurrent connections
- ✅ Robust data integrity with ACID compliance
- ✅ Superior performance for complex queries
- ✅ Production-ready with connection pooling
- ✅ JSON field support for future enhancements
- ✅ Optional pgvector extension for native vector operations

---

## Testing

### Run All Tests
```bash
python manage.py test
```

### Run Specific Tests
```bash
# Test retrieval functions
python manage.py test chat.tests.RetrievalTests

# Test MMR algorithm
python manage.py test chat.tests.MMRTests

# Test chunking
python manage.py test chat.tests.ChunkingTests
```

### Test Coverage
- ✅ Document chunking
- ✅ MMR algorithm (with/without diversity)
- ✅ Semantic search (with/without MMR)
- ✅ Document filtering
- ✅ Edge cases (empty corpus, no embeddings)

---

## Architecture

### System Design Overview

The chatbot follows a **modular, orchestrated architecture** using LangGraph to coordinate multiple specialized nodes in a workflow. This design ensures clean separation of concerns, testability, and extensibility.

### Core Components

#### 1. **Document Management** (`chat/models.py`)
- **`Document`**: Stores raw text with metadata
- **`DocumentChunk`**: Text chunks with 384-dim SBERT embeddings
- **Chunking Strategy**: Semantic paragraph-based splitting (500 chars max)
- **Storage**: PostgreSQL with indexed foreign keys

#### 2. **Embeddings & Retrieval** (`chat/embedding_utils.py`, `chat/retrieval.py`)
- **Model**: `all-MiniLM-L6-v2` (SBERT)
- **Dimensions**: 384-dimensional vectors
- **Similarity**: Cosine similarity
- **MMR Algorithm**: Maximal Marginal Relevance for diverse results
- **Session Filtering**: Only retrieves chunks from documents bound to the current session
- **Configurable**: `top_k`, `lambda_param` (relevance vs diversity)

#### 3. **Chat Memory Design** (`chat/langgraph/nodes/load_history.py`, `chat/models.py`)

**Two-tier memory system** for efficient context management:

**Short-term Memory (Token-bounded History):**
- Loads recent messages from `ChatMessage` table
- **Token Budget**: 3000 tokens (configurable via `CHAT_MAX_TOKENS_CONTEXT`)
- **Minimum Turns**: 6 turns preserved regardless of token count
- **Trimming Strategy**: Oldest-first removal while respecting minimum turns
- **Token Estimation**: Uses `tiktoken` for accurate OpenAI token counting
- **Purpose**: Maintains immediate conversational context

**Long-term Memory (Rolling Session Summary):**
- Stored in `ChatSession.long_term_summary` field
- **Update Frequency**: Every 5 assistant turns (configurable via `SUMMARY_INTERVAL_TURNS`)
- **Summary Generation**: LLM-powered abstractive summarization (300-600 tokens)
- **Prompt Injection**: Included at the top of every chat prompt
- **Purpose**: Preserves conversation essence while keeping prompts compact

**Memory Configuration** (`chatserver/settings.py`):
```python
MEMORY_CONFIG = {
    "max_tokens_context": 3000,      # Token budget for history
    "min_history_turns": 6,          # Minimum turns to preserve
    "summary_interval_turns": 5,     # Summarize every N turns
}
```

**Benefits:**
- ✅ Prevents context window overflow
- ✅ Maintains conversation coherence across long sessions
- ✅ Balances recency (short-term) with continuity (long-term)
- ✅ Configurable for different use cases

#### 4. **LangGraph Orchestration** (`chat/langgraph/`)

**Graph Nodes:**
1. **`load_history`**: Loads and trims conversation history (3000 token limit, 6 turn minimum)
2. **`decide_retrieve`**: Heuristic to determine if RAG is needed based on message content
3. **`retrieve`**: Performs semantic search with MMR (session-filtered)
4. **`synthesize`**: Generates LLM response (synchronous)
5. **`synthesize_stream`**: Generates LLM response (streaming)
6. **`summarize`**: Updates long-term session summary (every 5 assistant turns)

**Workflow:**
```
load_history → decide_retrieve → [retrieve?] → synthesize_stream → summarize
```

**Retrieval Decision Logic:**
The `decide_retrieve` node uses heuristics to determine if RAG is needed:
- **Question words**: what, how, why, when, where, who, explain, tell, describe, give, show, list, provide
- **Question marks**: Any message ending with "?"
- **Long messages**: More than 10 words
- **Document references**: Contains "document", "file", "source", "according to", "based on"
- **Excludes simple responses**: hi, hello, thanks, ok, yes, no

#### 5. **LLM Integration** (`chat/llm.py`)
- **Provider**: OpenAI API
- **Models**: `gpt-4o-mini` (default), configurable
- **Features**: Synchronous and streaming responses
- **Error Handling**: Authentication, rate limits, general errors
- **Token Counting**: `tiktoken` for accurate estimation
- **Security**: API key loaded from environment variables

#### 6. **Streaming (SSE)** (`chat/views.py`, `chat/langgraph/graph.py`)
- **Protocol**: Server-Sent Events (SSE)
- **Events**: `token` (deltas), `done` (completion), `error` (failures)
- **Persistence**: Full text saved on completion, partial on disconnect
- **Keep-alive**: Heartbeat for long-running connections
- **Manual Orchestration**: `run_graph_stream` bypasses compiled graph for streaming

### Design Rationale

#### Why LangGraph?
- **Explicit Control Flow**: Clear, debuggable orchestration vs black-box chains
- **Conditional Logic**: Dynamic RAG based on message content
- **State Management**: Shared state across nodes with type safety
- **Testability**: Each node is independently testable
- **Extensibility**: Easy to add new nodes (e.g., fact-checking, citations)

#### Why Two-tier Memory?
- **Context Window Limits**: LLMs have token limits (8K-128K)
- **Cost Optimization**: Fewer tokens = lower API costs
- **Performance**: Smaller prompts = faster responses
- **Coherence**: Summaries preserve long-term context without verbatim history

#### Why MMR (Maximal Marginal Relevance)?
- **Diversity**: Prevents redundant chunks from dominating results
- **Coverage**: Ensures broader information retrieval
- **Configurable**: `lambda_param` balances relevance vs diversity
- **Better RAG**: More diverse context = better LLM responses

#### Why PostgreSQL over SQLite?
- **Concurrency**: Handles multiple simultaneous connections
- **Integrity**: ACID compliance for data consistency
- **Performance**: Better query optimization and indexing
- **Production-ready**: Industry standard for web applications
- **Scalability**: Supports future growth (pgvector for native embeddings)

#### Why Server-Sent Events (SSE)?
- **Simplicity**: Easier than WebSockets for one-way streaming
- **HTTP-based**: Works with standard proxies and load balancers
- **Auto-reconnect**: Built-in browser reconnection
- **Progressive Enhancement**: Graceful fallback to synchronous API

---

## Tech Stack

- **Framework**: Django 5.0.6
- **API**: Django REST Framework 3.15
- **Database**: PostgreSQL 15 (with connection pooling)
- **Orchestration**: LangGraph (LangChain ecosystem)
- **LLM Provider**: OpenAI API (gpt-4o-mini)
- **Embeddings**: sentence-transformers (SBERT, all-MiniLM-L6-v2)
- **Token Counting**: tiktoken
- **Streaming**: Server-Sent Events (SSE)
- **CORS**: django-cors-headers
- **Environment**: python-dotenv
- **Python**: 3.10+

---

## Project Structure

```
chatserver/
├── chat/                    # Main app
│   ├── models.py           # Database models
│   ├── serializers.py      # DRF serializers
│   ├── views.py            # API endpoints
│   ├── urls.py             # URL routing
│   ├── retrieval.py        # Search & MMR
│   ├── embedding_utils.py  # Embedding generation
│   ├── chunking.py         # Text chunking
│   ├── llm.py              # OpenAI integration
│   ├── prompts.py          # Prompt engineering
│   ├── tests.py            # Unit tests (34 tests)
│   ├── langgraph/          # LangGraph orchestration (Phase 5)
│   │   ├── graph.py        # Graph definition & execution
│   │   ├── state.py        # Shared state schema
│   │   └── nodes/          # Individual graph nodes
│   │       ├── load_history.py
│   │       ├── decide_retrieve.py
│   │       ├── retrieve.py
│   │       ├── synthesize.py
│   │       ├── synthesize_stream.py  # Streaming variant
│   │       └── summarize.py
│   └── management/
│       └── commands/
│           └── ingest_docs.py
├── chatserver/             # Project settings
│   ├── settings.py
│   └── urls.py
├── static/
│   └── sse-demo.html       # Streaming demo
├── requirements.txt
├── manage.py
└── README.md
```

---

## 🌐 Live Demo

**Production Demo**: [https://191.101.81.150](https://191.101.81.150) ← **Access the live system here**  
**Local Development**: [http://localhost:8000/demo/](http://localhost:8000/demo/) (when running locally)

**Features:**
- 📄 **Document Upload**: Upload .txt files (max 500KB) and bind to chat sessions
- 🎯 **Session Management**: Independent sessions with document isolation
- ⚡ **Real-time Streaming**: See AI responses token-by-token as they're generated
- 🔍 **Smart Retrieval**: Automatic RAG triggering based on message content
- 💬 **Conversation History**: View full chat history in the UI
- 🔄 **Error Handling**: Graceful handling of connection issues
- 🎨 **Modern UI**: Clean, responsive design with clear workflow instructions

**How to Use:**
1. **Upload Document**: Choose a .txt file (max 500KB) and click "Upload & Process"
2. **Wait for Processing**: File is chunked and embedded (button turns green)
3. **Start Chat**: Click "🚀 Start Chat" to begin conversation
4. **Ask Questions**: Type questions about your document
5. **Reset**: Click "🔄 Reset Chat" to start over with a new document

**Technical Details:**
- Session-based document binding (one document per session)
- Automatic retrieval triggering for questions and long messages
- Token-bounded memory (3000 tokens, 6 turn minimum)
- Rolling summaries every 5 assistant turns
- Uses Fetch API with streaming response
- Handles `delta`, `done`, and `error` events

---

## Development

### Running Locally

1. Follow the [Quick Start](#quick-start) instructions
2. Access the local demo at: `http://localhost:8000/demo/`
3. Or use the API directly (see [API Endpoints](#api-endpoints))

### Upload and Test Documents
```bash
# Create a session first
curl -X POST http://localhost:8000/api/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Session"}'

# Upload document to session
curl -X POST http://localhost:8000/api/documents/upload/ \
  -F "file=@mydoc.txt" \
  -F "session_id=your-session-uuid"

# Test chat with automatic retrieval
curl -X POST http://localhost:8000/api/chat/stream/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-uuid",
    "message": "What is this document about?"
  }'
```

### Test Retrieval Logic
```bash
# Test session-specific retrieval
curl -X POST http://localhost:8000/api/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search query",
    "session_id": "your-session-uuid",
    "top_k": 5
  }'
```

### Admin Interface
```bash
# Access at http://localhost:8000/admin/
# View sessions, documents, chunks, and messages
# Username/password: Set with createsuperuser
```

---

## License

MIT
