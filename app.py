# app.py  –  Schema Therapy Bot  (Streamlit)
# Meets all core requirements + 2 medium + 1 hard optional tasks

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from costs import calculate_cost, format_cost
from rag import (
    advanced_retrieve,
    book_is_loaded,
    get_loaded_books,
    ingest_books_folder,
    needs_rebuild,
    read_index_stats,
)
from tools import find_technique, save_session, search_memory

load_dotenv()

# ── Security & validation ──────────────────────────────────────────────────────
MAX_MESSAGE_LENGTH   = 2000
MAX_MESSAGES_SESSION = 50
RATE_LIMIT_SECONDS   = 5

def validate_input(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Please enter a message."
    if len(text) > MAX_MESSAGE_LENGTH:
        return False, f"Message too long ({len(text)} chars). Keep it under {MAX_MESSAGE_LENGTH}."
    suspicious = [
        "ignore previous instructions", "ignore all instructions",
        "you are now", "disregard your instructions", "forget your system prompt",
    ]
    if any(p in text.lower() for p in suspicious):
        return False, "I noticed something unusual in your message. Please rephrase."
    return True, ""

def check_rate_limit() -> bool:
    last = st.session_state.get("last_message_time")
    if last is None:
        return True
    return (datetime.now() - last).total_seconds() >= RATE_LIMIT_SECONDS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Schema Therapy Bot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] {
    background-color: #f9f9f9 !important;
    border-right: 1px solid #e5e5e5 !important;
}
[data-testid="stSidebar"] * { color: #1a1a1a !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: #bbb !important;
    margin-bottom: 0.5rem !important;
}
.main .block-container {
    max-width: 740px !important;
    padding: 0 2rem 8rem !important;
    margin: 0 auto !important;
}

/* Title */
.app-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a1a1a;
    padding: 1.6rem 0 0.3rem;
    letter-spacing: -0.02em;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #999;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #efefef;
    margin-bottom: 1.6rem;
}

/* Messages */
.msg-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1.2rem;
    align-items: flex-start;
}
.msg-row.user { flex-direction: row-reverse; }
.avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 600;
    flex-shrink: 0; margin-top: 2px;
}
.avatar.bot  { background: #f0f0f0; color: #555; font-size: 0.9rem; }
.avatar.user { background: #1a1a1a; color: #fff; font-size: 0.68rem; }
.bubble {
    max-width: 84%;
    padding: 0.7rem 0.95rem;
    border-radius: 12px;
    font-size: 0.9rem;
    line-height: 1.65;
}
.bubble.bot {
    background: #f7f7f7;
    color: #1a1a1a;
    border: 1px solid #ececec;
    border-top-left-radius: 3px;
}
.bubble.user {
    background: #1a1a1a;
    color: #fff;
    border-top-right-radius: 3px;
}
.msg-meta {
    font-size: 0.68rem;
    color: #ccc;
    margin-top: 3px;
    margin-left: 40px;
}
.info-box {
    margin-top: 5px;
    margin-left: 40px;
    background: #fafafa;
    border: 1px solid #f0f0f0;
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
    font-size: 0.73rem;
    color: #888;
    line-height: 1.5;
}
.info-box b { color: #555; }

/* Input */
.stTextArea textarea {
    background: #fff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #999 !important;
    box-shadow: 0 0 0 2px rgba(0,0,0,0.05) !important;
}

/* Buttons */
.stButton > button {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 1px solid #ddd !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #f5f5f5 !important;
    border-color: #bbb !important;
}

/* Cost rows */
.cost-row {
    display: flex;
    justify-content: space-between;
    padding: 0.38rem 0;
    border-bottom: 1px solid #f2f2f2;
    font-size: 0.79rem;
}
.cost-label { color: #999; }
.cost-value { font-weight: 500; color: #1a1a1a; }

/* Pills */
.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
}
.pill-green { background: #f0faf0; color: #2d8a2d; border: 1px solid #c3e6c3; }
.pill-gray  { background: #f5f5f5; color: #999;    border: 1px solid #e5e5e5; }

hr { border-color: #f0f0f0 !important; margin: 0.85rem 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 4px; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Auto-load books from data/books/ folder ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def auto_load_books():
    """Load all PDFs from data/books/ on startup."""
    return ingest_books_folder()

with st.spinner("Loading knowledge base…"):
    book_load_result = auto_load_books()


# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "session_cost": 0.0,
        "session_tokens": 0,
        "last_prompt_cost": 0.0,
        "last_prompt_tokens": 0,
        "session_saved": False,
        "last_message_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── LLM ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=api_key)

llm          = get_llm()
tools_list   = [save_session, search_memory, find_technique]
llm_with_tools = llm.bind_tools(tools_list)

SYSTEM_PROMPT = """You are an expert Schema Therapy coach, trained in Jeffrey Young's model. \
You have read the user's books deeply. You are warm but direct — you explain, teach, and give \
practical help. You do NOT give generic therapy advice. Every answer should feel different and \
tailored to exactly what the person just said.

HOW TO RESPOND:
- Read what the person said carefully. Respond to THEIR specific situation, not a template.
- When relevant, name the Schema Mode at work (Vulnerable Child, Angry Child, Detached Protector, \
  Punitive Parent, Healthy Adult) and explain what it means FOR THEM specifically.
- When relevant, name the Schema driving it (Abandonment, Emotional Deprivation, Defectiveness, \
  Subjugation, Mistrust, etc.) and explain why they feel this way.
- Use the retrieved book passages to ground your answer. Quote or reference the books naturally.
- Vary your responses. Not every answer needs numbered steps. Sometimes a short explanation is \
  better. Sometimes a question. Sometimes a technique. Read the situation.
- If someone is in emotional pain, be human first — then explain. If someone asks a factual \
  question, answer it directly.

TOOLS — use precisely:
- find_technique: call when the user asks for exercises, techniques, or step-by-step practices in any form — "give me an exercise", "what exercises exist", "more exercises", "give me a technique for X", "duok pratimų", "daugiau pratimų". Do NOT call it for general questions about schemas or "what can you help me with". Call it at most ONCE per response.
- search_memory: call ONLY when the user explicitly references past sessions — "remember", "last time", "before", "previous session", "we talked about". Do not call it on the very first message.
- save_session: call when the user says goodbye, wants to end, or asks to save.

LANGUAGE RULE — CRITICAL:
Look at the user's LAST message only. Detect its language. Write your ENTIRE response in that language.
Lithuanian last message → entire response in Lithuanian.
English last message → entire response in English.
This applies even after tool calls — the tool result language does not matter, only the user's message language.
Never mix languages in one response.

ENDING RULE:
Never end your response with a question like "Would you like to explore this further?" or "Shall we discuss more?"
These feel robotic and disconnected. End with a statement, an insight, or an invitation to share more — but not a hollow question.

Security: Only discuss emotional wellbeing, schemas, relationships, and psychological patterns.

Book passages are provided below each message. Today's date: {date}""".format(date=datetime.now().strftime("%B %d, %Y"))


def run_agent(user_message: str) -> dict:
    """
    Run one agent turn. Returns a dict with:
    reply, cost, tokens, sources, tools_used, query_variants
    """
    # Advanced RAG: retrieve with query translation
    docs, query_variants = advanced_retrieve(user_message, k=6)

    sources = []
    passages = []
    for doc in docs:
        page      = doc.metadata.get("page", "?")
        book_name = doc.metadata.get("source_book", doc.metadata.get("source", "book"))
        passages.append(f"[{book_name} p.{page}] {doc.page_content}")
        sources.append({
            "page":      page,
            "book":      book_name,
            "snippet":   doc.page_content[:150] + "…",
        })

    # Build message history (last 10 turns)
    history = []
    for msg in st.session_state.messages[-10:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    augmented = user_message
    if passages:
        augmented += "\n\n[KNOWLEDGE BASE — retrieved automatically from indexed books, NOT provided by the user]:\n" + "\n\n---\n\n".join(passages)

    # Detect user language explicitly and inject into system prompt
    # This ensures tool results (which may be in a different language) don't pollute the response language
    def _detect_language(text: str) -> str:
        # Lithuanian special characters are 100% reliable when present
        if any(c in "ąčęėįšųūžĄČĘĖĮŠŲŪŽ" for c in text):
            return "Lithuanian"
        # Lithuanian words that don't exist in other languages
        lt_words = {
            "aš", "jis", "ji", "mes", "jūs", "yra", "buvo", "kaip", "ką",
            "labai", "taip", "dėl", "noriu", "galiu", "žinau", "kodėl",
            "ar", "prisimeni", "kokia", "mano", "spalva", "kas", "apie",
            "kada", "kuris", "kuri", "jeigu", "nes", "bet", "arba",
            "norėčiau", "gali", "negali", "nežinau", "suprantu", "jaučiu",
            "jaučiuosi", "manau", "galvoju", "sakau", "klausiu",
        }
        words = set(text.lower().replace("?", "").replace(".", "").replace(",", "").split())
        if len(words & lt_words) >= 1:
            return "Lithuanian"
        # Try langdetect for everything else
        try:
            from langdetect import detect
            code = detect(text)
            lang_names = {
                "lt": "Lithuanian", "en": "English", "de": "German",
                "fr": "French", "es": "Spanish", "it": "Italian",
                "pl": "Polish", "ru": "Russian", "pt": "Portuguese",
                "nl": "Dutch", "sv": "Swedish", "no": "Norwegian",
                "da": "Danish", "fi": "Finnish", "lv": "Latvian",
                "et": "Estonian", "uk": "Ukrainian", "cs": "Czech",
            }
            return lang_names.get(code, code)
        except Exception:
            return "English"

    detected_lang = _detect_language(user_message)
    lang_instruction = (
        f"\n\nCRITICAL LANGUAGE OVERRIDE: The user's current message is in {detected_lang}. "
        f"You MUST respond ENTIRELY in {detected_lang}. "
        f"The conversation history may be in a different language — ignore that. "
        f"Only the language of the CURRENT message matters. "
        f"Do NOT write even one sentence in any other language."
    )
    system_with_lang = SystemMessage(content=SYSTEM_PROMPT + lang_instruction)
    # Add a final reminder right before the user message so it's the last thing the LLM reads
    lang_reminder = SystemMessage(content=f"REMINDER: Respond in {detected_lang} only.")
    messages = [system_with_lang] + history + [lang_reminder, HumanMessage(content=augmented)]

    total_cost   = 0.0
    total_tokens = 0
    tools_used   = []

    response = llm_with_tools.invoke(messages)
    meta     = response.usage_metadata or {}
    in_tok   = meta.get("input_tokens", 0)
    out_tok  = meta.get("output_tokens", 0)
    total_tokens += in_tok + out_tok
    total_cost   += calculate_cost("gpt-4o-mini", in_tok, out_tok)

    if response.tool_calls:
        messages.append(response)
        from langchain_core.messages import ToolMessage
        tool_map = {t.name: t for t in tools_list}

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_fn   = tool_map.get(tool_name)
            tools_used.append(tool_name)

            if tool_fn:
                if tool_name == "save_session" and "conversation" not in tool_args:
                    tool_args = {"conversation": "\n".join(
                        f"{'Client' if m['role']=='user' else 'Therapist'}: {m['content']}"
                        for m in st.session_state.messages
                    )}
                try:
                    tool_result = tool_fn.invoke(tool_args)
                    if tool_name == "save_session":
                        st.session_state.session_saved = True
                except Exception as e:
                    tool_result = f"Tool error: {str(e)}"
            else:
                tool_result = "Tool not found."

            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))

        final    = llm_with_tools.invoke(messages)
        meta2    = final.usage_metadata or {}
        in2, out2 = meta2.get("input_tokens", 0), meta2.get("output_tokens", 0)
        total_tokens += in2 + out2
        total_cost   += calculate_cost("gpt-4o-mini", in2, out2)
        final_response = final.content
    else:
        final_response = response.content

    return {
        "reply":          final_response,
        "cost":           total_cost,
        "tokens":         total_tokens,
        "sources":        sources,
        "tools_used":     tools_used,
        "query_variants": query_variants,
    }


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Knowledge base status
    st.markdown("### Knowledge Base")
    loaded_books = get_loaded_books()
    if loaded_books:
        for b in loaded_books:
            st.markdown(
                f'<span class="pill pill-green">✓</span> '
                f'<span style="font-size:0.78rem;color:#555">{b[:36]}</span>',
                unsafe_allow_html=True,
            )
        st.caption(f"{len(loaded_books)} book(s) in data/books/")
    else:
        st.markdown('<span class="pill pill-gray">No books in data/books/</span>', unsafe_allow_html=True)
        st.caption("Add PDF files to the data/books/ folder.")

    if st.button("🔄 Update Book Index", use_container_width=True):
        if loaded_books:
            with st.spinner("Indexing books… this may take a minute."):
                try:
                    result = ingest_books_folder(force=True)
                    st.cache_resource.clear()
                    st.success(f"✓ {result.get('books', 0)} book(s) · {result.get('chunks', 0)} chunks indexed.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        else:
            st.warning("No PDFs found in data/books/")

    stats = read_index_stats()
    if stats:
        with st.expander("Index details", expanded=False):
            for book, chunks in stats.items():
                st.caption(f"{book[:40]}: {chunks} chunks")

    st.markdown("---")
    st.markdown("### Session")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save & End", use_container_width=True):
            if st.session_state.messages:
                with st.spinner("Saving…"):
                    # Skip the welcome message (first assistant message with no prior user message)
                    real_msgs = [m for m in st.session_state.messages
                                 if not (m["role"] == "assistant" and "I'm glad you're here" in m.get("content", ""))]
                    if not real_msgs:
                        st.warning("Nothing meaningful to save yet.")
                        st.stop()
                    conv   = "\n".join(
                        f"{'Client' if m['role']=='user' else 'Therapist'}: {m['content']}"
                        for m in real_msgs
                    )
                    result = save_session.invoke({"conversation": conv})
                    st.session_state.session_saved = True
                    # Show save confirmation then clear chat — session is truly ended
                    st.session_state.messages = [{"role": "assistant", "content": result + "\n\n---\n*Session ended. Click **New Session** to start a new conversation.*"}]
                st.rerun()
            else:
                st.warning("Nothing to save.")
    with col2:
        if st.button("New Session", use_container_width=True):
            for key in ["messages", "session_cost", "session_tokens",
                        "last_prompt_cost", "last_prompt_tokens",
                        "session_saved", "last_message_time"]:
                st.session_state[key] = [] if key == "messages" else (0.0 if "cost" in key or "tokens" in key else None if key == "last_message_time" else False)
            st.rerun()

    st.markdown("---")
    st.markdown("### Usage & Cost")
    st.markdown(f"""
    <div class="cost-row">
        <span class="cost-label">Last message</span>
        <span class="cost-value">{st.session_state.last_prompt_tokens:,} tokens · {format_cost(st.session_state.last_prompt_cost)}</span>
    </div>
    <div class="cost-row" style="border:none">
        <span class="cost-label">This session</span>
        <span class="cost-value">{st.session_state.session_tokens:,} tokens · {format_cost(st.session_state.session_cost)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Past Sessions")
    sessions_dir = Path("data/sessions")
    if sessions_dir.exists():
        session_files = sorted(sessions_dir.glob("*.json"), reverse=True)[:6]
        if session_files:
            for sf in session_files:
                try:
                    data = json.loads(sf.read_text())
                    ts   = data.get("timestamp", "")[:8]
                    if len(ts) == 8:
                        ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:]}"
                    with st.expander(data.get("title", sf.stem)[:50]):
                        st.caption(ts)
                        st.write(data.get("summary", ""))
                except Exception:
                    pass
        else:
            st.caption("No saved sessions yet.")
    else:
        st.caption("No saved sessions yet.")




# ── MAIN CHAT ──────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🧠 Schema Therapy Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Evidence-based schema therapy · Multilingual · Grounded in your books</div>', unsafe_allow_html=True)

# Session length warning
msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
if msg_count >= MAX_MESSAGES_SESSION:
    st.warning(f"You've sent {msg_count} messages. Consider saving and starting a new session.")

# Welcome
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello. I'm glad you're here.\n\n"
            "This is a space to explore what's on your mind, at whatever pace feels right. "
            "I work within schema therapy — interested in the patterns and experiences that shape "
            "how you feel and relate to others.\n\n"
            "You can write in any language and I'll respond in yours. "
            "What would you like to talk about today?"
        ),
    })

# Render messages using native Streamlit chat — proper markdown rendering
for msg in st.session_state.messages:
    role       = msg["role"]
    tokens_info = msg.get("tokens_info", "")
    sources     = msg.get("sources", [])
    tools_used  = msg.get("tools_used", [])

    with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
        st.markdown(msg["content"])

        if role == "assistant":
            if tokens_info:
                st.caption(tokens_info)
            if sources:
                source_str = " · ".join(f"*{s['book']}* p.{s['page']}" for s in sources)
                st.caption(f"📖 Sources: {source_str}")
            if tools_used:
                st.caption(f"🔧 Tools: {', '.join(tools_used)}")

# Input — st.chat_input supports Enter key natively
if not book_is_loaded():
    st.caption("⚠️ No books found. Add PDFs to data/books/ and restart.")
if st.session_state.get("session_saved"):
    st.info("Session ended. Click **New Session** in the sidebar to start a new conversation.")
    user_input = None
else:
    user_input = st.chat_input("Write a message…")
if user_input and user_input.strip():
    now = datetime.now()
    last = st.session_state.get("last_message_time")
    if last is not None:
        elapsed = (now - last).total_seconds()
        remaining = RATE_LIMIT_SECONDS - elapsed
        if remaining > 0:
            secs = int(remaining) + 1
            st.warning(f"⏳ Please wait **{secs} second{'s' if secs != 1 else ''}** before sending another message.")
            st.stop()

    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        st.error(error_msg)
        st.stop()
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    # Render user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input.strip())

    # Run agent and render response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking…"):
            try:
                result = run_agent(user_input.strip())
            except Exception as e:
                result = {
                    "reply": f"Something went wrong: {str(e)}",
                    "cost": 0.0, "tokens": 0,
                    "sources": [], "tools_used": [], "query_variants": [],
                }
        st.markdown(result["reply"])
        tokens_info = f"{result['tokens']:,} tokens · {format_cost(result['cost'])}"
        st.caption(tokens_info)
        if result["sources"]:
            source_str = " · ".join(f"*{s['book']}* p.{s['page']}" for s in result["sources"])
            st.caption(f"📖 Sources: {source_str}")
        if result["tools_used"]:
            st.caption(f"🔧 Tools: {', '.join(result['tools_used'])}")

    # Stamp time AFTER response is done — rate limit counts from here
    st.session_state.last_message_time = datetime.now()

    # Update session state
    st.session_state.last_prompt_cost   = result["cost"]
    st.session_state.last_prompt_tokens = result["tokens"]
    st.session_state.session_cost      += result["cost"]
    st.session_state.session_tokens    += result["tokens"]

    st.session_state.messages.append({
        "role":           "assistant",
        "content":        result["reply"],
        "tokens_info":    tokens_info,
        "sources":        result["sources"],
        "tools_used":     result["tools_used"],
        "query_variants": result["query_variants"],
    })

    # Rerun so sidebar token counts update
    st.rerun()