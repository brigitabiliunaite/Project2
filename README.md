# Specialised Chatbot — Advanced RAG & Tool Calling

## Task description

Create a specialised chatbot that leverages advanced RAG techniques and tool calling to provide domain-specific assistance. The goal is to build something that could be valuable in a real-world context. We will be using LangChain, your framework (Streamlit or Next.js) of choice, and implementing advanced RAG techniques. The intended code editor for this project is **VS Code**.

Your chatbot will:
1. Implement advanced RAG with query translation and structured retrieval
2. Include tool calling capabilities for practical tasks
3. Focus on a specific domain or use case
4. Provide detailed, context-aware responses

Example use cases (but feel free to create your own):
- Career Consultant Bot: Uses World Economic Forum reports and job market data to provide career advice
- Technical Documentation Assistant: Helps developers understand and work with specific frameworks or libraries
- Financial Advisor Bot: Analyses market trends and provides investment insights
- Healthcare Information Assistant: Provides accurate medical information from verified sources
- Legal Research Assistant: Helps with legal queries using case law and legal documents

---

## My chosen use case

**Schema Therapy Coaching Bot** — a conversational AI chatbot grounded in schema therapy literature. Schema therapy, developed by Jeffrey Young, is a psychotherapy model focused on identifying deep emotional patterns (schemas) formed in childhood that affect adult life. The bot answers questions, identifies schemas and modes, suggests practical exercises, and remembers past sessions — all based on real schema therapy books indexed from PDF. It makes evidence-based psychological knowledge accessible in a conversational, personalised way.

---

## Task requirements

### Core requirements

| # | Requirement | Status |
|---|---|---|
| 1 | RAG Implementation — knowledge base, embeddings, chunking, similarity search | ✅ |
| 2 | Tool Calling — at least 3 relevant tool calls | ✅ 3 tools |
| 3 | Domain Specialisation — focused knowledge base, domain prompts, security | ✅ |
| 4 | Technical Implementation — LangChain, error handling, logging, validation, rate limiting, API key management | ✅ |
| 5 | User Interface — Streamlit, sources, tool call results, progress indicators | ✅ |

### Implemented optional tasks

**Easy**
- ✅ Easy #1 — Conversation history: sessions saved to JSON, searchable in future conversations
- ✅ Easy #3 — Source citations: every response shows book name and page number

**Medium**
- ✅ Medium #2 — Real-time knowledge base updates: SHA256 fingerprint detection + Update Index button
- ✅ Medium #5 — Token usage and cost display: shown per message and as session total

**Hard**
- ✅ Hard #6 — Multi-language support: automatic per-message language detection using `langdetect` (55 languages), with Lithuanian character fallback

---

## 🧠 Schema Therapy Bot

A conversational AI chatbot built with Streamlit and LangChain, grounded in schema therapy literature. The bot answers questions, suggests practical exercises, and remembers past sessions — all based on real schema therapy books indexed from PDF.

### What it does

- Answers questions about schema therapy, early maladaptive schemas, and psychological modes using content retrieved directly from indexed books
- Suggests practical step-by-step exercises grounded in the literature
- Remembers past therapy sessions and references them in future conversations
- Responds in whatever language the user writes in (55 languages via `langdetect`)
- Tracks token usage and dollar cost per message and per session
- Saves sessions with an AI-generated title, summary, and key themes

---

### Tech stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector database | ChromaDB (local) |
| LLM framework | LangChain |
| Monitoring | LangSmith |
| PDF loading | LangChain `PyPDFLoader` |
| Language detection | `langdetect` |

---

### Project structure

```
app.py          — Streamlit UI, agent loop, security, session management
rag.py          — PDF ingestion, chunking, embedding, advanced retrieval
tools.py        — 3 LangChain tools: save_session, search_memory, find_technique
costs.py        — Token counting and cost calculation

data/
  books/              — schema therapy PDF books (add your own here)
  sessions/           — saved session JSON files (auto-created)
  vectorstore/        — ChromaDB book embeddings (auto-created on first run)
  memory_vectorstore/ — ChromaDB session memory embeddings (auto-created)

.env            — API keys (see Setup)
```

---

### Setup

**Requirements:** Python 3.11

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install streamlit langchain langchain-openai langchain-community \
    langchain-chroma chromadb pypdf python-dotenv langsmith \
    langchain-text-splitters langdetect
```

**Create a `.env` file** in the project root:

```
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=schema-therapy-bot
```

**Add your books:** drop schema therapy PDF files into `data/books/`.

**Run:**
```bash
streamlit run app.py
```

The app indexes all books automatically on first startup. To add books later, click the **🔄 Update Book Index** button in the sidebar.

---

### How RAG works

RAG (Retrieval Augmented Generation) means the bot answers questions using your actual books, not just what the AI was trained on.

1. **Chunking** — each PDF is split into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`
2. **Embedding** — each chunk is converted to a vector using OpenAI `text-embedding-3-small` and stored in ChromaDB
3. **Query translation** — the user's message is rewritten into 3 schema therapy search variants by the LLM (e.g. "I feel alone" → `["emotional deprivation schema", "unmet attachment needs", "social isolation coping patterns"]`)
4. **MMR retrieval** — each variant searches ChromaDB using Maximal Marginal Relevance, returning diverse results from different parts of different books
5. **Context injection** — retrieved chunks are injected into the LLM prompt so the answer is grounded in the books
6. **Source citations** — each response shows the book name and page number of every chunk used

---

### The 3 tools

| Tool | Triggered when | What it does |
|---|---|---|
| `find_technique` | User asks for an exercise or step-by-step practice | Searches books for exercises, generates a focused technique |
| `save_session` | User clicks Save & End or says goodbye | LLM generates title/summary/themes, saves JSON, embeds into memory |
| `search_memory` | User references a past session | Searches the memory vectorstore for relevant past conversation chunks |

Tools are registered with `llm.bind_tools()`. The LLM reads their docstring descriptions and decides autonomously when to call each one. Results are passed back as `ToolMessage` objects before the final response.

---

### Security

- **Input length limit** — messages over 2000 characters are rejected
- **Prompt injection detection** — phrases like `"ignore previous instructions"` are blocked before reaching the LLM
- **Rate limiting** — 15-second cooldown between messages
- **Domain restriction** — system prompt restricts the LLM to schema therapy topics only
- **API key management** — all keys loaded from `.env` via `python-dotenv`, never hardcoded

---

### Monitoring and logging

All interactions are traced in LangSmith (`smith.langchain.com`, project: `schema-therapy-bot`). Each trace shows the full message chain, which tools were called, token counts per step, and latency. Terminal logs show retrieval statistics after each message.

---

### Session persistence

When a session is saved:
1. LLM generates a `title`, `summary`, and `key_themes`
2. JSON file saved to `data/sessions/TIMESTAMP_TITLE.json` with full conversation text
3. Conversation embedded into memory vectorstore for future retrieval
4. Appears in the sidebar under **Past Sessions**

---

### Optional tasks in detail

**Hard #6 — Multi-language support**
Per-message language detection using the `langdetect` library (supports 55 languages). The detected language is injected as an explicit override into the system prompt for every message, ensuring the response language stays consistent even when tool results come back in a different language. Falls back to Lithuanian character detection if `langdetect` fails on very short or ambiguous input.

**Medium #5 — Token usage and cost tracking**
Every response shows token count and dollar cost. The sidebar shows session totals. Calculated in `costs.py` using OpenAI's published pricing for `gpt-4o-mini` (input: $0.000150/1K, output: $0.000600/1K).

**Medium #2 — Real-time knowledge base updates**
SHA256 fingerprinting of each file's name, size, and modification timestamp detects when books change. The sidebar shows a warning when the index is stale. The **🔄 Update Book Index** button rebuilds in-place using batched embedding (100 chunks per request) to stay within OpenAI payload limits.

**Easy #3 — Source citations**
Every response displays `📖 Sources: *Book Name* p.83 · p.134` — book name and page number for each retrieved chunk.

**Easy #1 — Conversation history**
Sessions persist as JSON files on disk and are searchable via the memory vectorstore in future sessions.

---

### Potential improvements

- **Streaming responses** — display text word by word as it generates (`llm.stream()` + `st.write_stream()`)
- **RAGAs evaluation** — objectively measure retrieval quality (Hard task #9)
- **Always-on memory search** — search memory on every message with a similarity threshold rather than keyword triggering
