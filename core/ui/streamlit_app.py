"""Streamlit web UI for ROBOTS Personal Assistant â€” Multi-conversation support."""
from __future__ import annotations

import streamlit as st

from core.agent import PersonalAssistantAgent
from core.settings import settings


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Agent singleton
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@st.cache_resource
def preload_agent() -> PersonalAssistantAgent:
    return PersonalAssistantAgent()


def get_agent() -> PersonalAssistantAgent:
    if "assistant_agent" not in st.session_state:
        st.session_state.assistant_agent = preload_agent()
    return st.session_state.assistant_agent


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Conversation management
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def init_conversations() -> None:
    """Initialize conversation state in session."""
    agent = get_agent()
    memory = agent.memory

    if "conversations" not in st.session_state:
        st.session_state.conversations = memory.list_conversations()

    if "active_conv_id" not in st.session_state:
        active_id = memory.get_active_conversation_id()
        if not active_id:
            active_id = memory.create_conversation()
            st.session_state.conversations = memory.list_conversations()
        st.session_state.active_conv_id = active_id

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def refresh_conversations() -> None:
    """Reload conversation list from memory."""
    agent = get_agent()
    st.session_state.conversations = agent.memory.list_conversations()


def load_chat_history() -> None:
    """Load conversation messages from memory into session_state chat_history."""
    agent = get_agent()
    messages = agent.memory.get_message_history()
    st.session_state.chat_history = []
    for msg in messages:
        for part in msg.parts:
            kind = getattr(part, "part_kind", "")
            content = getattr(part, "content", "")
            if kind == "user-prompt" and content:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": str(content),
                    "tool_events": [],
                    "latency_ms": {},
                })
            elif kind == "text" and content:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": str(content),
                    "tool_events": [],
                    "latency_ms": {},
                })


def switch_to_conversation(conv_id: str) -> None:
    """Switch to a different conversation."""
    agent = get_agent()
    if agent.memory.switch_conversation(conv_id):
        st.session_state.active_conv_id = conv_id
        load_chat_history()
        st.rerun()


def create_new_conversation() -> None:
    """Create a new conversation."""
    agent = get_agent()
    conv_id = agent.memory.create_conversation()
    st.session_state.active_conv_id = conv_id
    st.session_state.chat_history = []
    refresh_conversations()
    st.rerun()


def delete_conversation(conv_id: str) -> None:
    """Delete a conversation."""
    agent = get_agent()
    agent.memory.delete_conversation(conv_id)
    refresh_conversations()
    st.session_state.active_conv_id = agent.memory.get_active_conversation_id()
    st.session_state.chat_history = []
    st.rerun()


def rename_conversation(conv_id: str) -> None:
    """Rename a conversation (via text input in sidebar)."""
    key = f"rename_{conv_id}"
    new_name = st.session_state.get(key, "").strip()
    if new_name:
        agent = get_agent()
        agent.memory.rename_conversation(conv_id, new_name)
        refresh_conversations()
        st.rerun()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# UI Rendering
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _render_custom_css() -> None:
    """Inject custom CSS for modern UI."""
    st.markdown("""
    <style>
    /* â”€â”€ Global â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    .stApp { background: #0f0f0f; }
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    section[data-testid="stSidebar"] {
        background: #1a1a2e;
        border-right: 1px solid #2a2a4a;
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #a78bfa; }

    /* â”€â”€ Chat messages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    .stChatMessage {
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        padding: 0.75rem 1rem;
    }
    .stChatMessage[data-testid="stChatMessage"]:has(div:contains("user")) {
        background: #1e3a5f;
        border-color: #2563eb;
    }
    .stChatMessage[data-testid="stChatMessage"]:has(div:contains("assistant")) {
        background: #1a1a2e;
        border-color: #7c3aed;
    }

    /* â”€â”€ Chat input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    .stChatInput textarea {
        background: #1a1a2e;
        border: 1px solid #3a3a5a;
        border-radius: 12px;
        color: #e0e0e0;
    }

    /* â”€â”€ Buttons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    .stButton button {
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3);
    }

    /* â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    [data-testid="stMetricValue"] { color: #a78bfa; }

    /* â”€â”€ Expanders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    .streamlit-expanderHeader {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
    }

    /* â”€â”€ Scrollbar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f0f0f; }
    ::-webkit-scrollbar-thumb { background: #3a3a5a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
    </style>
    """, unsafe_allow_html=True)


def _fix_mojibake_text(text: str) -> str:
    """Repair text that was saved as UTF-8 but decoded as Windows-1252."""
    if not isinstance(text, str) or not any(mark in text for mark in ("Ã", "Â", "á»", "áº", "ð", "â", "Æ", "Ä")):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text


def render_sidebar() -> None:
    """Render sidebar with conversation list + settings."""
    with st.sidebar:
        # â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("### **ROBOTS** Assistant")
        st.caption("Trợ lý AI cá nhân - đa cuộc trò chuyện")

        st.divider()

        # â”€â”€ New Chat Button â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if st.button("Cuộc trò chuyện mới", use_container_width=True, type="primary"):
            create_new_conversation()

        st.divider()

        # â”€â”€ Conversation List â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("##### Lịch sử trò chuyện")

        for conv in st.session_state.conversations:
            conv_id = conv["id"]
            conv_name = _fix_mojibake_text(conv["name"])
            is_active = conv_id == st.session_state.active_conv_id

            cols = st.columns([5, 1, 1])

            # Conversation button
            btn_label = conv_name
            clicked = cols[0].button(
                btn_label,
                key=f"switch_{conv_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                disabled=is_active,
            )
            if clicked and not is_active:
                switch_to_conversation(conv_id)

            # Rename button
            if cols[1].button("Sửa", key=f"edit_{conv_id}"):
                st.session_state[f"show_rename_{conv_id}"] = True

            # Delete button
            if cols[2].button("Xóa", key=f"del_{conv_id}"):
                delete_conversation(conv_id)

            # Rename input
            if st.session_state.get(f"show_rename_{conv_id}", False):
                with st.form(key=f"rename_form_{conv_id}"):
                    new_name = st.text_input(
                        "Tên mới:",
                        value=conv_name,
                        key=f"rename_{conv_id}",
                    )
                    col_save, col_cancel = st.columns(2)
                    if col_save.form_submit_button("Lưu"):
                        rename_conversation(conv_id)
                    if col_cancel.form_submit_button("Hủy"):
                        st.session_state[f"show_rename_{conv_id}"] = False
                        st.rerun()

        st.divider()

        # â”€â”€ Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with st.expander("Cấu hình"):
            st.write(f"Model: **{settings.model}**")
            st.write(f"Profile: {settings.runtime_profile}")
            st.write(f"Max tokens: {settings.llm_max_completion_tokens}")

        with st.expander("Tools"):
            st.markdown(
                "- **get_weather**: Thời tiết\n"
                "- **web_search**: Tìm kiếm realtime\n"
                "- **calculate**: Tính toán\n"
                "- **Music tools**: Phát / Dừng / Tắt nhạc\n"
                "- **Các tools khác**"
            )


def push_history(role: str, content: str, tool_events: list | None = None, latency_ms: dict | None = None) -> None:
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "tool_events": tool_events or [],
        "latency_ms": latency_ms or {},
    })


def handle_text_prompt(agent: PersonalAssistantAgent, prompt: str) -> None:
    # Auto-rename conversation from first message
    conv_id = st.session_state.active_conv_id
    agent.memory.auto_rename_first_message(conv_id, prompt)

    push_history("user", prompt)
    response = agent.run(prompt)
    push_history(
        "assistant",
        response.text,
        tool_events=response.tool_events,
        latency_ms=response.latency_ms,
    )
    refresh_conversations()


def render_chat_history() -> None:
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(_fix_mojibake_text(item["content"]))
            if item["role"] == "assistant":
                if item.get("latency_ms"):
                    cols = st.columns(3)
                    labels = [("tool", "Tool"), ("llm", "LLM"), ("end_to_end", "Tổng")]
                    idx = 0
                    for key, label in labels:
                        if key in item["latency_ms"]:
                            cols[idx].metric(label, f"{item['latency_ms'][key]} ms")
                            idx += 1
                if item.get("tool_events"):
                    with st.expander("Tool events"):
                        for event in item["tool_events"]:
                            st.code(event, language=None)


def main() -> None:
    st.set_page_config(
        page_title="ROBOTS Personal Assistant",
        page_icon="R",
        layout="wide",
    )

    _render_custom_css()

    # Initialize
    init_conversations()

    agent = get_agent()

    # Sidebar
    render_sidebar()

    # Main header
    active_conv = next(
        (c for c in st.session_state.conversations if c["id"] == st.session_state.active_conv_id),
        None,
    )
    if active_conv:
        st.markdown(f"### {_fix_mojibake_text(active_conv['name'])}")

    # Chat history
    render_chat_history()

    # Chat input
    prompt = st.chat_input("Nhập câu hỏi hoặc yêu cầu...")
    if prompt:
        with st.spinner("Đang xử lý..."):
            handle_text_prompt(agent, prompt)
        st.rerun()
