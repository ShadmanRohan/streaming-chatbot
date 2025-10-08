# ChatServer — RAG-Powered Chatbot Backend

Django 5 backend with document ingestion, semantic search (with MMR), and chat capabilities.

## Features

- 📄 **Document Management**: Upload and process text documents
- 🔍 **Semantic Search**: SBERT embeddings with cosine similarity
- 🎯 **MMR (Maximal Marginal Relevance)**: Diverse, relevant retrieval results
- 💬 **Chat Sessions**: Conversation management with message history
- 🧪 **Comprehensive Tests**: Unit tests for all retrieval functions

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda activate legal_assistant
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create superuser (for admin access)
python manage.py createsuperuser
```

### 2. Run Server

```bash
python manage.py runserver
```

### 3. Health Check

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
Upload and process a document.

```bash
curl -X POST http://localhost:8000/api/documents/upload/ \
  -F "file=@document.txt" \
  -F "auto_process=true"
```

**Response:**
```json
{
  "id": "uuid-here",
  "filename": "document.txt",
  "chunk_count": 5,
  "auto_processed": true,
  "created_at": "2025-10-08T12:00:00Z"
}
```

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

**Filter by document:**
```bash
curl -X POST http://localhost:8000/api/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search text",
    "top_k": 5,
    "document_ids": ["doc-uuid-1", "doc-uuid-2"]
  }'
```

**Parameters:**
- `query` (required): Search query text
- `top_k` (optional, default: 3): Number of results to return
- `use_mmr` (optional, default: true): Enable MMR for diversity
- `lambda_param` (optional, default: 0.5): MMR trade-off (0=diversity, 1=relevance)
- `document_ids` (optional): Filter by specific document IDs

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

#### POST `/api/chat/send/`
Send a message and get AI response with RAG (Retrieval-Augmented Generation).

**Request:**
```bash
curl -X POST http://localhost:8000/api/chat/send/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-uuid",
    "message": "What is machine learning?",
    "retrieve": true,
    "top_k": 3,
    "use_mmr": true,
    "model": "gpt-4o-mini"
  }'
```

**Parameters:**
- `session_id` (required): Chat session UUID
- `message` (required): User's message/question
- `retrieve` (optional, default: true): Enable RAG retrieval
- `top_k` (optional, default: 3): Number of chunks to retrieve
- `use_mmr` (optional, default: true): Use MMR for diverse results
- `lambda_param` (optional, default: 0.5): MMR trade-off parameter
- `model` (optional, default: gpt-4o-mini): OpenAI model to use

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
```

**Important:** The chat endpoint requires a valid OpenAI API key. Get one from [OpenAI Platform](https://platform.openai.com/api-keys).

### Database

**SQLite (default):**
- Auto-configured, works out of the box
- Uses `db.sqlite3` file

**PostgreSQL (optional):**
- Set `DB_NAME` env var to enable
- Requires `psycopg2-binary` (already in requirements)
- For pgvector support, install extension in Postgres

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

### Components

1. **Document Management** (`chat/models.py`)
   - `Document`: Raw text storage
   - `DocumentChunk`: Text chunks with embeddings

2. **Chunking** (`chat/chunking.py`)
   - Semantic paragraph-based splitting
   - Configurable max length

3. **Embeddings** (`chat/embedding_utils.py`)
   - SBERT model: `all-MiniLM-L6-v2`
   - 384-dimensional vectors
   - Cosine similarity

4. **Retrieval** (`chat/retrieval.py`)
   - Basic similarity search
   - MMR for diversity
   - Document filtering

5. **Chat** (`chat/models.py`)
   - `ChatSession`: Conversation container
   - `ChatMessage`: User/assistant messages

---

## Tech Stack

- **Framework**: Django 5.0.6
- **API**: Django REST Framework 3.15
- **Embeddings**: sentence-transformers (SBERT)
- **Database**: SQLite (default) / PostgreSQL (optional)
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
│   ├── tests.py            # Unit tests
│   └── management/
│       └── commands/
│           └── ingest_docs.py
├── chatserver/             # Project settings
│   ├── settings.py
│   └── urls.py
├── requirements.txt
├── manage.py
└── README.md
```

---

## Development

### Add New Document
```bash
# Via API
curl -X POST http://localhost:8000/api/documents/upload/ \
  -F "file=@mydoc.txt"

# Via admin
# Go to http://localhost:8000/admin/chat/document/
```

### Test Retrieval
```bash
# Search across all documents
curl -X POST http://localhost:8000/api/retrieve/ \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 5}'
```

### Admin Interface
```bash
# Access at http://localhost:8000/admin/
# Username/password: Set with createsuperuser
```

---

## Next Steps (Roadmap)

- [x] Phase 0-3: Project setup, models, retrieval, MMR ✅
- [x] Phase 4: Chat endpoint with LLM integration ✅
- [ ] Phase 5: LangGraph orchestration  
- [ ] Phase 6: Streaming responses (SSE)
- [ ] Phase 7: Session memory & summarization
- [ ] Phase 8: Rate limiting, CORS, observability
- [ ] Phase 9: Documentation & deployment

---

## License

MIT
