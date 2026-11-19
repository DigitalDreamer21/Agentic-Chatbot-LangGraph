import os
import tempfile
import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command

from backend import chatbot, get_all_threads, ingest_rag_document


# ----------------------------- Helpers -----------------------------

def extract_text(content: Any) -> str:
    """Convert LangChain string or multimodal content blocks to readable text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        if isinstance(text_value, dict):
            return extract_text(text_value)
        return ""

    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            block_text = extract_text(block)
            if block_text:
                parts.append(block_text)
        return "".join(parts)

    return ""


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_hitl"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id: str):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", []) if state and state.values else []


def get_pending_interrupt(thread_id: str):
    """Return the first pending LangGraph interrupt, if one exists."""
    try:
        snapshot = chatbot.get_state(
            {"configurable": {"thread_id": thread_id}}
        )

        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0]

        for task in (getattr(snapshot, "tasks", ()) or ()):
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]
    except Exception:
        # A thread with no checkpoint has no pending interrupt.
        return None

    return None


def save_pending_interrupt(thread_id: str, interrupt_object) -> None:
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id: str) -> None:
    pending = get_pending_interrupt(thread_id)
    if pending is not None:
        save_pending_interrupt(thread_id, pending)
    else:
        current = st.session_state.get("pending_hitl")
        if current and current.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def stream_assistant_text(input_value, config, status_holder):
    """Stream only readable assistant text; do not render tool-call payloads."""
    for chunk, metadata in chatbot.stream(
        input_value,
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage) or getattr(chunk, "type", None) == "tool":
            tool_name = getattr(chunk, "name", None) or "tool"
            status_box = status_holder.get("box")
            if status_box is None:
                status_holder["box"] = st.status(
                    f"🔧 Using `{tool_name}` …",
                    expanded=True,
                )
            else:
                status_box.update(
                    label=f"🔧 Using `{tool_name}` …",
                    state="running",
                    expanded=True,
                )
            continue

        # LangGraph message streaming normally emits AIMessageChunk objects.
        if isinstance(chunk, (AIMessage, AIMessageChunk)) or getattr(chunk, "type", None) == "ai":
            text_part = extract_text(getattr(chunk, "content", None))
            if text_part:
                yield text_part


def resume_hitl_execution(decision: str) -> None:
    pending = st.session_state.get("pending_hitl")
    if not pending:
        st.warning("There is no pending action to approve or reject.")
        return

    thread_id = pending["thread_id"]
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "hitl_resume_trace",
    }
    status_holder = {"box": None}

    try:
        with st.chat_message("assistant"):
            with st.status("🔄 Resuming the requested action...", expanded=True) as status:
                status_holder["box"] = status
                response = st.write_stream(
                    stream_assistant_text(
                        Command(resume=decision),
                        config,
                        status_holder,
                    )
                )

                next_interrupt = get_pending_interrupt(thread_id)
                if next_interrupt is not None:
                    save_pending_interrupt(thread_id, next_interrupt)
                    status.update(
                        label="⚠️ Another approval is required",
                        state="complete",
                        expanded=False,
                    )
                else:
                    st.session_state["pending_hitl"] = None
                    status.update(
                        label="✅ Action completed",
                        state="complete",
                        expanded=False,
                    )

        if response:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": response}
            )
        st.rerun()

    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")


# ----------------------------- Page setup -----------------------------

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
:root {
    --ink: #182238;
    --muted: #64748b;
    --brand: #5b5bd6;
    --brand-dark: #4545b8;
    --surface: #ffffff;
    --line: #e7eaf2;
}
.stApp {
    background: linear-gradient(135deg, #f4f7ff 0%, #f8f7ff 48%, #f2f8ff 100%);
    color: var(--ink);
}
[data-testid="stHeader"] {
    background: rgba(244,247,255,.88);
}
[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}
h1 {
    color: #182238 !important;
    font-weight: 800 !important;
    letter-spacing: -1.1px;
    padding-bottom: .35rem;
}
h2, h3, h4, p, label, li {
    color: #26334b !important;
}

/* Force readable text in all Streamlit markdown, including generated tables. */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] blockquote {
    color: #26334b !important;
}
[data-testid="stMarkdownContainer"] table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    overflow: hidden;
    border: 1px solid #dfe5f2;
    border-radius: 12px;
    background: #ffffff !important;
}
[data-testid="stMarkdownContainer"] thead th {
    background: #edf1ff !important;
    color: #202b49 !important;
    font-weight: 700;
    border-bottom: 1px solid #dfe5f2 !important;
}
[data-testid="stMarkdownContainer"] tbody td {
    background: #ffffff !important;
    color: #26334b !important;
    border-bottom: 1px solid #edf0f6 !important;
}
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    padding: .7rem .8rem !important;
    text-align: left;
    vertical-align: top;
}
[data-testid="stMarkdownContainer"] tr:last-child td {
    border-bottom: 0 !important;
}
[data-testid="stMarkdownContainer"] a {
    color: #4b4bc4 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef1ff 0%, #f7f8ff 100%);
    border-right: 1px solid #dfe4f4;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #202b49 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #64708a;
}
.stButton > button {
    border: 1px solid #d9def0;
    border-radius: 11px;
    background: #ffffff;
    color: #34415e;
    font-weight: 600;
    min-height: 2.55rem;
    box-shadow: 0 2px 7px rgba(38, 53, 95, .05);
    transition: all .18s ease;
}
.stButton > button:hover {
    border-color: #7777e8;
    color: #4545b8;
    background: #f7f7ff;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #5b5bd6, #7777e8);
    border: 0;
    color: white;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.85);
    border: 1px dashed #aeb8dc;
    border-radius: 14px;
    padding: .55rem;
}
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    color: #26334b !important;
    border: 1px solid #e7eaf2;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin: .8rem 0;
    box-shadow: 0 5px 18px rgba(39, 54, 92, .045);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
    color: #26334b !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #eef2ff;
    border-color: #dce3ff;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
}
[data-testid="stChatInput"] {
    border: 1px solid #dce2f1;
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(45, 58, 100, .09);
}
[data-testid="stChatInput"] textarea {
    color: #1f2a44 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8993a8 !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: white;
    border-radius: 12px;
}
[data-testid="stAlert"], [data-testid="stStatusWidget"] {
    border-radius: 13px;
}
hr {
    border-color: #dfe4f2;
}
code {
    color: #4b4bc4 !important;
    background: #f0f1ff !important;
    border-radius: 5px;
}
small, .stCaption {
    color: #77839a !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic Chatbot with LangGraph")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])
sync_pending_interrupt(st.session_state["thread_id"])


# ----------------------------- Sidebar -----------------------------

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ----------------------------- PDF Knowledge Base -----------------------------

st.sidebar.divider()
st.sidebar.subheader("📚 PDF Knowledge Base")
st.sidebar.caption("Upload PDFs here to add them to your chatbot's searchable knowledge base.")

knowledge_pdfs = st.sidebar.file_uploader(
    "Choose PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    key="knowledge_base_pdfs",
)

if st.sidebar.button(
    "Process and index PDFs",
    disabled=not knowledge_pdfs,
    use_container_width=True,
    key="index_knowledge_pdfs",
):
    for uploaded_pdf in knowledge_pdfs:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_pdf.getvalue())
                temporary_path = temp_file.name

            with st.sidebar.status(
                f"Processing {uploaded_pdf.name}...",
                expanded=True,
            ) as pdf_status:
                ingest_rag_document(temporary_path)
                pdf_status.update(
                    label=f"✅ {uploaded_pdf.name} indexed successfully",
                    state="complete",
                    expanded=False,
                )
        except Exception as error:
            st.sidebar.error(
                f"Could not process {uploaded_pdf.name}: {error}"
            )
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    st.sidebar.success("PDF processing finished. You can now ask questions in the chat.")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id), key=f"thread_{thread_id}", use_container_width=True):
        st.session_state["thread_id"] = thread_id
        saved_messages = load_conversation(thread_id)
        history = []

        for message in saved_messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            text_value = extract_text(message.content)
            if text_value:
                history.append({"role": role, "content": text_value})

        st.session_state["message_history"] = history
        sync_pending_interrupt(thread_id)
        st.rerun()


# ----------------------------- Chat history -----------------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(extract_text(message["content"]))


# ----------------------------- Human approval -----------------------------

pending_hitl = st.session_state.get("pending_hitl")
current_thread_has_pending_hitl = bool(
    pending_hitl
    and pending_hitl.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )
    approve_col, reject_col = st.columns(2)

    with approve_col:
        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True,
        ):
            resume_hitl_execution("yes")

    with reject_col:
        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True,
        ):
            resume_hitl_execution("no")


# ----------------------------- Chat input and PDF upload -----------------------------

submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl,
)

if submission:
    user_input = (submission.text or "").strip()
    uploaded_files = submission.files or []

    # Ingest attached PDFs before sending the user's accompanying question.
    for uploaded_pdf in uploaded_files:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_pdf.getvalue())
                temporary_path = temp_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(temporary_path)

            st.toast(f"{uploaded_pdf.name} indexed successfully.", icon="✅")
        except Exception as error:
            st.error(f"PDF processing failed for {uploaded_pdf.name}: {error}")
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    if user_input:
        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        config = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "chat_trace",
        }
        status_holder = {"box": None}

        try:
            with st.chat_message("assistant"):
                response_text = st.write_stream(
                    stream_assistant_text(
                        {"messages": [HumanMessage(content=user_input)]},
                        config,
                        status_holder,
                    )
                )

                pending_interrupt = get_pending_interrupt(
                    st.session_state["thread_id"]
                )

                if pending_interrupt is not None:
                    save_pending_interrupt(
                        st.session_state["thread_id"],
                        pending_interrupt,
                    )
                    response_text = (
                        (response_text or "")
                        + "\n\n⚠️ This stock purchase requires your approval. "
                        "Use the Approve Purchase or Reject Purchase button below."
                    )

                status_box = status_holder.get("box")
                if status_box is not None:
                    if pending_interrupt is not None:
                        status_box.update(
                            label="⏸️ Waiting for human approval",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status_box.update(
                            label="✅ Tool finished",
                            state="complete",
                            expanded=False,
                        )

            st.session_state["message_history"].append(
                {"role": "assistant", "content": response_text or "I couldn't generate a text response. Please try again."}
            )

            if pending_interrupt is not None:
                st.rerun()

        except RuntimeError as error:
            if "cannot schedule new futures after shutdown" in str(error).lower():
                st.error(
                    "The model/client executor was shut down while streaming. "
                    "Restart Streamlit and retry. If it repeats, share the full "
                    "terminal traceback; this runtime error depends on the installed "
                    "Google GenAI/Streamlit versions and cannot safely be hidden by retrying."
                )
            else:
                st.error(f"Chatbot runtime error: {error}")
        except Exception as error:
            st.error(f"Chatbot execution failed: {error}")
