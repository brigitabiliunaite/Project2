# tools.py  –  The 3 LangChain tools for the Schema Therapy Bot

import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

SESSIONS_DIR     = Path("data/sessions")
MEMORY_DB_PATH   = "data/memory_vectorstore"
VECTORSTORE_PATH = "data/vectorstore"   # must match rag.py

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _get_memory_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="session_memory",
        embedding_function=embeddings,
        persist_directory=MEMORY_DB_PATH,
    )


# ── Tool 1: save_session ───────────────────────────────────────────────────────

@tool
def save_session(conversation: str) -> str:
    """
    Save the current therapy session to disk and embed it into memory.
    Call this when the user says goodbye, wants to save, or clicks Save & End.
    Input: the full conversation text.
    """
    if not conversation or not conversation.strip():
        return "Nothing to save — the conversation is empty."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = f"""You are a professional psychotherapist using schema therapy.
Read the session below and return ONLY valid JSON (no markdown, no code fences) with these keys:
- "title": short session title, max 8 words
- "summary": 3-5 sentence clinical summary as a therapist's session note
- "key_themes": list of 3-5 schema therapy themes (e.g. "abandonment schema", "vulnerable child mode")

SESSION:
{conversation[:6000]}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if model ignores instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        notes = json.loads(raw)
    except json.JSONDecodeError:
        notes = {
            "title": f"Session {datetime.now().strftime('%Y-%m-%d')}",
            "summary": raw[:500],
            "key_themes": [],
        }

    # Save full session to JSON file
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in " _-" else "" for c in notes["title"])[:50]
    filepath   = SESSIONS_DIR / f"{timestamp}_{safe_title}.json"

    filepath.write_text(json.dumps({
        "timestamp":   timestamp,
        "title":       notes["title"],
        "summary":     notes["summary"],
        "key_themes":  notes["key_themes"],
        "conversation": conversation,
    }, ensure_ascii=False, indent=2))

    # Embed summary + full conversation chunks into memory vectorstore
    memory_store = _get_memory_store()
    summary_text = (
        f"Session: {notes['title']}\n"
        f"Date: {datetime.now().strftime('%B %d, %Y')}\n"
        f"Summary: {notes['summary']}\n"
        f"Themes: {', '.join(notes['key_themes'])}"
    )
    chunk_size  = 1500
    conv_chunks = [conversation[i:i+chunk_size] for i in range(0, len(conversation), chunk_size)]
    all_texts   = [summary_text] + conv_chunks
    all_meta    = [{"timestamp": timestamp, "title": notes["title"], "type": "summary"}] + [
        {"timestamp": timestamp, "title": notes["title"], "type": "full_text", "chunk": i}
        for i in range(len(conv_chunks))
    ]
    memory_store.add_texts(texts=all_texts, metadatas=all_meta)

    return (
        f"✅ Session saved as **\"{notes['title']}\"**\n\n"
        f"📋 **Summary:** {notes['summary']}\n\n"
        f"🔖 **Themes:** {', '.join(notes['key_themes'])}"
    )


# ── Tool 2: search_memory ──────────────────────────────────────────────────────

@tool
def search_memory(query: str) -> str:
    """
    Search past therapy session notes to find relevant history.
    Call proactively when the user references the past, or when connecting
    current themes to earlier patterns across sessions.
    Input: short description of what you are looking for.
    """
    try:
        memory_store = _get_memory_store()
        results = memory_store.similarity_search(query, k=3)
    except Exception:
        return "No past sessions found yet."

    if not results:
        return "No relevant past session notes found on this topic."

    parts = ["**Relevant memories from past sessions:**\n"]
    for i, doc in enumerate(results, 1):
        parts.append(f"**Memory {i}:**\n{doc.page_content}\n")
    return "\n".join(parts)


# ── Tool 3: find_technique ─────────────────────────────────────────────────────

@tool
def find_technique(situation: str) -> str:
    """
    Search the schema therapy books for a practical exercise or technique.
    Call when the user seems stuck, asks what to do, or when a concrete
    book-based exercise would genuinely help right now.
    Input: brief description of what the client is experiencing.
    """
    if not os.path.exists(VECTORSTORE_PATH) or not os.listdir(VECTORSTORE_PATH):
        return "No books are indexed yet. Add PDFs to data/books/ and restart."

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    book_store = Chroma(
        collection_name="book_knowledge",
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_PATH,
    )

    results = book_store.similarity_search(
        f"exercise technique coping strategy for: {situation}", k=4
    )

    if not results:
        return "No specific technique found in the books for this situation."

    context = "\n\n---\n\n".join(doc.page_content for doc in results)
    llm     = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    response = llm.invoke(
        f"""You are a schema therapist. Based ONLY on these book excerpts, describe one clear 
practical exercise the client can do right now. Be step-by-step and specific.
Mention this comes from the book. Respond in the same language as the situation below.

CLIENT SITUATION: {situation}

BOOK EXCERPTS:
{context}"""
    )
    return f"📖 **Technique from the book:**\n\n{response.content}"