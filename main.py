import io
import charset_normalizer
import fitz
import streamlit as st
from decouple import config
from openai import OpenAI
import json
import os
import tiktoken
import uuid
from datetime import datetime
import glob
import re
from typing import List, Dict, Union, Optional

# Constants with API-enforced limits
CHAT_NAME_MAX_LENGTH = 50
CHAT_DESC_MAX_LENGTH = 100
MAX_API_TOKENS = 131072  # 128K tokens (128 * 1024 = 131072)
CHUNK_SIZE = 32000
MAX_FILE_CONTEXT_LENGTH = 100000  # Reduced to leave room for chat history
# Ensure we have an absolute path for chat history
CHAT_HISTORY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "chat_history")
)
# Updated models for V3.1
MODELS = ["deepseek-chat", "deepseek-reasoner"]
MAX_OUTPUT_TOKENS_CHAT = 8192  # 8K maximum for deepseek-chat
MAX_OUTPUT_TOKENS_REASONER = 65536  # 64K maximum for deepseek-reasoner

API_KEY = config("DEEPSEEK_API_KEY", default="")
encoding = tiktoken.encoding_for_model("gpt-4")

# Default context
DEFAULT_CONTEXT = "You are a helpful assistant. Provide clear and concise answers. If you are writing a code make sure to summarize and provide a concise code, optimize the code output to the smartest and the shortest way with better readability and functionality"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
DEFAULT_PROMPT_FILE = "default_prompt.json"

os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
if not os.path.exists(CHAT_HISTORY_DIR):
    try:
        os.makedirs(CHAT_HISTORY_DIR)
    except Exception as e:
        st.error(f"Could not create chat history directory: {e}")


def start_new_chat():
    """Start fresh chat session without saving current chat."""
    reset_chat_state()
    st.rerun()  # Force UI update


def reset_chat_state():
    """Completely reset all chat-related session state."""
    st.session_state.update(
        {
            "chat_history": [],
            "file_context": [],
            "system_message": DEFAULT_CONTEXT,
            "messages": [],
            "reasoning_content": None,  # New for reasoning model
        }
    )


def save_chat_session(chat_history, chat_name, chat_desc) -> str:
    """Save current chat session with provided name/description"""
    try:
        if not chat_history or not isinstance(chat_history, list):
            st.sidebar.error(f"Invalid chat history: empty or invalid format")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        session_id = str(uuid.uuid4())[:8]

        chat_dir_abs = os.path.abspath(CHAT_HISTORY_DIR)
        filename = os.path.join(chat_dir_abs, f"chat_{timestamp}_{session_id}.json")
        temp_path = f"{filename}.tmp"

        data = {
            "metadata": {
                "name": chat_name[:CHAT_NAME_MAX_LENGTH],
                "description": chat_desc[:CHAT_DESC_MAX_LENGTH] if chat_desc else "",
                "created_at": timestamp,
                "session_id": session_id,
            },
            "chat_history": chat_history,
            "default_prompt": st.session_state.get("default_prompt", ""),
            "file_context": st.session_state.get("file_context", []),
            "system_message": st.session_state.get("system_message", DEFAULT_CONTEXT),
            "reasoning_content": st.session_state.get("reasoning_content", None),  # Save reasoning
        }

        os.makedirs(chat_dir_abs, exist_ok=True)
        if not os.path.exists(chat_dir_abs):
            st.sidebar.error(f"Failed to create directory: {chat_dir_abs}")
            return ""

        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)

        if not os.path.exists(temp_path):
            st.sidebar.error(f"Failed to create temporary file: {temp_path}")
            return ""

        os.rename(temp_path, filename)

        if os.path.exists(filename):
            st.session_state["refresh_chats_flag"] = True
            return filename
        else:
            st.sidebar.error("Failed to save chat")
            return ""

    except Exception as e:
        st.sidebar.error(f"Error saving chat: {str(e)}")
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return ""


def save_chat_modal():
    """Reusable modal for saving chats with name/description"""
    st.sidebar.markdown("### Save Chat")
    st.sidebar.caption("Please enter a name for your chat")

    with st.sidebar.form(key="save_chat_form", clear_on_submit=True):
        chat_name = st.text_input(
            "Chat Name",
            max_chars=CHAT_NAME_MAX_LENGTH,
            value=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        chat_desc = st.text_area(
            "Description (Optional)",
            max_chars=CHAT_DESC_MAX_LENGTH,
            help="Add a brief description to help identify this chat later",
        )
        submitted = st.form_submit_button("Save Chat")

        if submitted:
            if not chat_name:
                st.sidebar.error("No chat name provided")
            else:
                return chat_name, chat_desc

    return None, None

def validate_prompt(prompt_text: str) -> bool:
    """Validate prompt content meets basic requirements"""
    if not isinstance(prompt_text, str):
        return False
    if len(prompt_text.strip()) == 0:
        return False
    if len(prompt_text) > 10000:  # Reasonable upper limit
        return False
    return True

def load_default_prompt() -> str:
    """Load default prompt from file"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), DEFAULT_PROMPT_FILE)

        if not os.path.exists(file_path):
            return ""

        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read().strip()

        # Always expect JSON format for consistency
        try:
            data = json.loads(content)
            return data.get("prompt", "")
        except json.JSONDecodeError:
            st.error("Default prompt file must be valid JSON")
            return ""

    except Exception as e:
        st.error(f"Error loading default prompt: {e}")
        return ""


def save_default_prompt(prompt_text):
    """Save default prompt with validation"""
    if not validate_prompt(prompt_text):
        st.error("Invalid prompt content")
        return False

    try:
        file_path = os.path.join(os.path.dirname(__file__), DEFAULT_PROMPT_FILE)

        # Save as JSON for consistency
        data = {"prompt": prompt_text, "updated": datetime.now().isoformat()}

        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        st.error(f"Error saving default prompt: {e}")
        return False

def load_chat_session(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return {
            "metadata": data.get(
                "metadata",
                {
                    "name": os.path.basename(filename),
                    "description": "",
                    "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                },
            ),
            "chat_history": data.get("chat_history", []),
            "default_prompt": data.get("default_prompt", ""),
            "file_context": data.get("file_context", []),
            "system_message": data.get("system_message", DEFAULT_CONTEXT),
            "reasoning_content": data.get("reasoning_content", None),  # Load reasoning
            "id": data.get("id", str(uuid.uuid4())),
            "timestamp": data.get(
                "timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
        }


def list_saved_chats():
    """List all saved chat sessions"""
    try:
        chat_dir_abs = os.path.abspath(CHAT_HISTORY_DIR)

        if not os.path.exists(chat_dir_abs):
            os.makedirs(chat_dir_abs, exist_ok=True)
            return []

        pattern = os.path.join(chat_dir_abs, "*.json")

        if not os.access(chat_dir_abs, os.R_OK):
            st.sidebar.error(f"Cannot access chat history directory")
            return []

        files = glob.glob(pattern)

        valid_files = []
        for file in files:
            if os.path.isfile(file) and os.access(file, os.R_OK):
                try:
                    with open(file, "rb") as f:
                        f.read(10)
                    valid_files.append(file)
                except Exception:
                    pass

        files = valid_files

        return sorted(
            files,
            key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0,
            reverse=True,
        )

    except Exception as e:
        st.sidebar.error(f"Error listing chats: {str(e)}")
        return []


def enforce_token_limit(messages):
    """Ensure total tokens stay within API limits"""
    total_tokens = calculate_context_usage(messages)

    if total_tokens <= MAX_API_TOKENS:
        return messages

    system_message = messages[0]
    chat_history = messages[1:]

    system_tokens = calculate_context_usage([system_message])
    remaining_tokens = MAX_API_TOKENS - system_tokens

    while True:
        chat_history = chat_history[-20:]
        history_tokens = calculate_context_usage(chat_history)

        if system_tokens + history_tokens <= MAX_API_TOKENS:
            break

        for msg in chat_history:
            current_tokens = len(encoding.encode(msg["content"]))
            if current_tokens > 1000:
                msg["content"] = smart_truncate(msg["content"], current_tokens // 2)

    return [system_message] + chat_history


def smart_truncate(text: str, max_tokens: int, is_code: bool = False) -> str:
    """Optimized truncation for both text and code with structure preservation."""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    if is_code:
        return truncate_code(text, max_tokens)

    return truncate_text(text, max_tokens)


def truncate_code(code: str, max_tokens: int) -> str:
    """Specialized code truncation that preserves structure."""
    patterns = [
        (r"^(import|from)\s", 100),
        (r"^(class|def)\s", 90),
        (r"^@", 80),
        (r"^#\s*[A-Z]", 70),
        (r"^#", 50),
        (r".+", 10),
    ]

    lines = code.split("\n")
    scored_lines = []

    for line in lines:
        score = 0
        for pattern, pattern_score in patterns:
            if re.match(pattern, line):
                score = pattern_score
                break
        scored_lines.append((score, line))

    scored_lines.sort(key=lambda x: (-x[0], lines.index(x[1])))

    output = []
    current_tokens = 0
    for score, line in scored_lines:
        line_tokens = len(encoding.encode(line))
        if current_tokens + line_tokens > max_tokens:
            continue
        output.append(line)
        current_tokens += line_tokens

    if current_tokens < max_tokens * 0.9:
        for line in lines:
            if line not in output:
                line_tokens = len(encoding.encode(line))
                if current_tokens + line_tokens <= max_tokens:
                    output.append(line)
                    current_tokens += line_tokens

    truncated = "\n".join(output)
    final_tokens = len(encoding.encode(truncated))

    if final_tokens > max_tokens:
        return encoding.decode(encoding.encode(truncated)[:max_tokens])

    if len(output) < len(lines):
        truncated += f"\n\n...[TRUNCATED {len(lines) - len(output)} LINES]..."

    return truncated


def truncate_text(text: str, max_tokens: int) -> str:
    """Optimized text truncation with better boundary detection."""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    mid_point = max_tokens // 2
    paragraphs = text.split("\n\n")

    if len(paragraphs) > 1:
        current_len = 0
        split_index = 0
        for i, para in enumerate(paragraphs):
            para_tokens = len(encoding.encode(para))
            if current_len + para_tokens > mid_point:
                split_index = i
                break
            current_len += para_tokens

        first_half = "\n\n".join(paragraphs[:split_index])
        second_half = "\n\n".join(paragraphs[split_index:])

        first_tokens = len(encoding.encode(first_half))
        remaining = max(0, max_tokens - first_tokens)

        if remaining > 100:
            second_tokens = len(encoding.encode(second_half))
            if second_tokens > remaining:
                second_half = encoding.decode(encoding.encode(second_half)[:remaining])

            truncated = f"{first_half}\n\n...[TRUNCATED]...\n\n{second_half}"
        else:
            truncated = first_half
    else:
        truncated = encoding.decode(tokens[:max_tokens])

    return truncated


def read_large_file(file) -> str:
    """Optimized file reader with better code detection and chunking."""
    # Updated code_extensions with new file types
    code_extensions = {
        "py": "python",
        "js": "javascript",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "h": "c-header",
        "sh": "bash",
        "php": "php",
        "rb": "ruby",
        "go": "go",
        "rs": "rust",
        "ts": "typescript",
        "md": "markdown",
        "html": "html",
        "css": "css",
        "mk": "makefile",          # Makefiles
        "xml": "xml",              # XML files
        "cfg": "ini",              # Configuration files
        "json": "json",            # JSON files
        "service": "ini",          # Systemd service files
        "nmconnection": "ini",     # NetworkManager connection files
        "j2": "jinja2",            # Jinja2 templates
        "conf": "apache",          # Apache/configuration files
    }

    ext = file.name.split(".")[-1].lower()
    is_code = ext in code_extensions

    try:
        if file.type == "application/pdf":
            doc = fitz.open(stream=io.BytesIO(file.read()), filetype="pdf")
            return "".join(page.get_text() for page in doc)

        content = []
        file.seek(0)

        # Replace chardet with charset_normalizer
        sample = file.read(1024)
        file.seek(0)

        # Use charset_normalizer instead of chardet
        detection_result = charset_normalizer.detect(sample)
        encoding_type = detection_result.get('encoding', 'utf-8') or 'utf-8'

        chunk_size = CHUNK_SIZE * (10 if is_code else 1)

        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            content.append(chunk.decode(encoding_type, errors="replace"))

        full_text = "".join(content)
        return smart_truncate(full_text, MAX_FILE_CONTEXT_LENGTH, is_code)

    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return ""


def get_user_balance():
    """Fetch user balance from DeepSeek API"""
    try:
        if not API_KEY:
            st.sidebar.warning("API key not configured")
            return None

        response = client._client.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("is_available", False):
            st.sidebar.warning("Balance information not available")
            return None

        return data.get("balance_infos", [{}])[0]
    except Exception as e:
        st.sidebar.error(f"Failed to fetch balance: {e}")
        return None


def calculate_context_usage(messages):
    """Calculate total token usage"""
    return sum(len(encoding.encode(msg["content"])) for msg in messages)


def update_system_message():
    """Build system message from components"""
    components = [DEFAULT_CONTEXT]

    # Add default prompt if exists
    if st.session_state["default_prompt"]:
        components.append(f"Additional Instructions:\n{st.session_state['default_prompt']}")

    # Add file context if exists
    if st.session_state["file_context"]:
        file_section = ["Uploaded Files:"]
        for file in st.session_state["file_context"]:
            truncated = smart_truncate(file["content"], MAX_FILE_CONTEXT_LENGTH // 3)
            file_section.append(f"File: {file['name']}\nContent:\n{truncated}")
        components.append("\n".join(file_section))

    # Combine and truncate
    st.session_state["system_message"] = smart_truncate(
        "\n\n".join(components), 
        MAX_FILE_CONTEXT_LENGTH
    )


def init_session_state():
    """Initialize all session state keys with proper order"""
    # Load default prompt FIRST
    default_prompt_content = load_default_prompt()

    # Initialize ALL session state defaults in one go
    defaults = {
        "chat_history": [],
        "default_prompt": default_prompt_content,
        "file_context": [],
        "temperature": 0.1,
        "refresh_chats_flag": True,
        "save_chat_clicked": False,
        "save_chat_bottom_clicked": False,
        "chat_save_path": None,
        "balance_info": None,
        "reasoning_content": None,
        "model_select": "deepseek-chat",
        "cache_metrics": {"hit": 0, "miss": 0, "efficiency": 0.0}
    }

    # Set all defaults that aren't already in session state
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Build system message after everything is loaded (this replaces the old logic)
    if "system_message" not in st.session_state:
        update_system_message()


# Initialize all session state values
init_session_state()

# Streamlit UI
st.set_page_config(
    initial_sidebar_state="auto", layout="wide", page_title="DeepSeek Chat V3.1"
)

# Sidebar
with st.sidebar:
    st.subheader("Chat Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🆕 New Chat", help="Start fresh chat (does not save current chat)"
        ):
            start_new_chat()
    with col2:
        if "save_chat_clicked" not in st.session_state:
            st.session_state.save_chat_clicked = False

        if (
            st.button("💾 Save Chat", key="save_chat_top", help="Save current chat")
            or st.session_state.save_chat_clicked
        ):
            if not st.session_state["chat_history"]:
                st.error("No chat messages to save!")
            else:
                st.session_state.save_chat_clicked = True
                st.sidebar.markdown("---")
                st.sidebar.markdown("## 💾 Save Your Chat")
                chat_name, chat_desc = save_chat_modal()
                if chat_name:
                    with st.spinner("Saving chat..."):
                        filename = save_chat_session(
                            st.session_state["chat_history"], chat_name, chat_desc
                        )
                        if filename:
                            st.success(f"Saved as {os.path.basename(filename)}")
                            st.session_state.save_chat_clicked = False
                            st.session_state["refresh_chats_flag"] = True
                            st.rerun()
    st.title("Settings")

    # Context usage display at top
    if st.session_state["chat_history"]:
        usage = calculate_context_usage(
            [{"role": "system", "content": st.session_state["system_message"]}]
            + st.session_state["chat_history"]
        )
        st.progress(min(usage / MAX_API_TOKENS, 1.0))
        st.caption(f"Context usage: {usage:,}/{MAX_API_TOKENS:,} tokens")

    # NEW: Cache efficiency display for V3.1
    if st.session_state["cache_metrics"]["hit"] > 0 or st.session_state["cache_metrics"]["miss"] > 0:
        st.subheader("Cache Efficiency")
        efficiency = st.session_state["cache_metrics"]["efficiency"]
        st.metric("Cache Efficiency", f"{efficiency:.1%}")
        st.caption(f"Hit: {st.session_state['cache_metrics']['hit']:,} tokens")
        st.caption(f"Miss: {st.session_state['cache_metrics']['miss']:,} tokens")

    st.subheader("Account Balance")
    if st.button("💰 Refresh Balance"):
        st.session_state.balance_info = get_user_balance()

    if st.session_state.balance_info:
        balance = st.session_state.balance_info
        st.metric(
            "Total Balance",
            f"{balance.get('total_balance', 0)} {balance.get('currency', 'USD')}",
        )
        st.metric(
            "Granted",
            f"{balance.get('granted_balance', 0)} {balance.get('currency', 'USD')}",
        )
        st.metric(
            "Topped Up",
            f"{balance.get('topped_up_balance', 0)} {balance.get('currency', 'USD')}",
        )
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    elif st.session_state.balance_info is None:
        st.info("Click 'Refresh Balance' to check your balance")

    # File upload - UPDATED with new file types
    uploaded_files = st.file_uploader(
        "Upload files",
        type=[
            "pdf", "txt", "py", "md", "csv", "json", "docx", "h", "ino", "sh", 
            "php", "js", "html", "cmd", "map", "c", "j2", "conf", "css",
            "mk", "xml", "cfg", "service", "nmconnection"  # Added new file types
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} files..."):
            new_files = []
            for file in uploaded_files:
                with st.expander(f"Processing {file.name}", expanded=False):
                    content = read_large_file(file)
                    if content:
                        new_files.append({"name": file.name, "content": content})
                        st.code(
                            f"Loaded {len(content.splitlines())} lines", language="text"
                        )

            st.session_state["file_context"] = new_files
            update_system_message()

    if st.session_state.get("file_context"):
        if st.button(
            "🗑️ Delete All Files",
            key="delete_all_files",
            help="Remove all uploaded files at once",
        ):
            st.session_state["file_context"] = []
            update_system_message()
            st.success("All uploaded files have been removed!")
            st.rerun()

    # Chat history section
    st.header("📚 Chat History", divider="rainbow")

    if "refresh_chats_flag" not in st.session_state:
        st.session_state["refresh_chats_flag"] = True

    saved_chats = list_saved_chats()

    with st.container():
        if not saved_chats:
            st.warning("📝 No saved chats found. Save a chat to see it here.")
            with st.expander("How to save a chat?"):
                st.write("1. Have a conversation in the main chat area")
                st.write("2. Click '💾 Save Chat' at the top or bottom of the sidebar")
                st.write("3. Enter a name and optional description")
                st.write("4. Click 'Save Chat' to store it")
                st.write("5. Your saved chat will appear here")
        else:
            chat_options = []
            chat_details = {}

            for i, chat_file in enumerate(saved_chats):
                try:
                    with open(chat_file, "r") as f:
                        data = json.load(f)

                    name = data.get("metadata", {}).get(
                        "name", os.path.basename(chat_file)
                    )
                    desc = data.get("metadata", {}).get("description", "")
                    created_at = data.get("metadata", {}).get("created_at", "")

                    if created_at:
                        try:
                            date_display = datetime.strptime(
                                created_at[:8], "%Y%m%d"
                            ).strftime("%b %d, %Y")
                        except Exception:
                            date_display = (
                                created_at[:8] if len(created_at) >= 8 else created_at
                            )
                    else:
                        date_display = "Unknown date"

                    display_name = f"{name} ({date_display})"
                    message_count = len(data.get("chat_history", []))

                    chat_options.append(display_name)
                    chat_details[display_name] = {
                        "filename": chat_file,
                        "description": desc,
                        "date": created_at,
                        "original_name": name,
                        "message_count": message_count,
                    }

                except Exception as e:
                    file_name = os.path.basename(chat_file)
                    display_name = f"{file_name} (Error loading)"
                    chat_options.append(display_name)
                    chat_details[display_name] = {
                        "filename": chat_file,
                        "description": f"Error: {str(e)}",
                        "date": "",
                        "original_name": file_name,
                        "message_count": 0,
                    }

            if st.button("🔄 Refresh Chat List", key="refresh_chat_list_btn"):
                st.session_state["refresh_chats_flag"] = True
                st.rerun()

            if chat_options:
                st.markdown("### Select a saved chat")
                selected_name = st.selectbox(
                    "Available chats:",
                    chat_options,
                    key="saved_chat_selector",
                    format_func=lambda x: f"{x} ({chat_details[x].get('message_count', 0)} messages)"
                    if x in chat_details
                    else x,
                )

                if selected_name:
                    with st.container():
                        if chat_details[selected_name]["description"]:
                            st.info(chat_details[selected_name]["description"])

                        if chat_details[selected_name].get("message_count", 0) > 0:
                            st.caption(
                                f"Contains {chat_details[selected_name]['message_count']} messages"
                            )

                        st.caption(
                            f"File: {os.path.basename(chat_details[selected_name]['filename'])}"
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(
                                "📥 Load Chat",
                                key="load_chat_btn",
                                use_container_width=True,
                            ):
                                try:
                                    with st.spinner(
                                        f"Loading {chat_details[selected_name]['original_name']}..."
                                    ):
                                        data = load_chat_session(
                                            chat_details[selected_name]["filename"]
                                        )
                                        st.session_state.update(
                                            {
                                                "chat_history": data["chat_history"],
                                                "default_prompt": data["default_prompt"],
                                                "file_context": data["file_context"],
                                                "system_message": data["system_message"],
                                                "reasoning_content": data.get("reasoning_content", None),
                                            }
                                        )
                                        st.success(
                                            f"Loaded: {chat_details[selected_name]['original_name']}"
                                        )
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to load chat: {e}")
                        with col2:
                            if st.button(
                                "🗑️ Delete",
                                key="delete_chat_btn",
                                use_container_width=True,
                            ):
                                try:
                                    full_path = chat_details[selected_name]["filename"]
                                    os.remove(full_path)
                                    st.success(
                                        f"Deleted {chat_details[selected_name]['original_name']}"
                                    )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting file: {e}")

    # Advanced Settings
    with st.expander("Advanced Settings"):
        # Model selection - UPDATED for V3.1
        model_choice = st.selectbox(
            "Model", 
            MODELS, 
            index=0, 
            key="model_select",
            help="deepseek-chat: Standard mode | deepseek-reasoner: Thinking mode with reasoning"
        )

        # NEW: Reasoning content display for deepseek-reasoner
        if st.session_state["model_select"] == "deepseek-reasoner":
            if st.session_state.get("reasoning_content"):
                with st.expander("View Reasoning Content", expanded=False):
                    st.text_area(
                        "Reasoning Chain",
                        value=st.session_state["reasoning_content"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )

        # Task selection
        task_type = st.selectbox(
            "Task Type",
            ["Coding/Math", "Normal Questions", "Data Analysis", "Creative Writing"],
            index=0,
            key="task_type_select",
        )

        # Dynamic temperature - UPDATED for reasoning model constraints
        temp_ranges = {
            "Coding/Math": (0.0, 0.3),
            "Data Analysis": (0.3, 0.7),
            "Normal Questions": (0.5, 0.9),
            "Creative Writing": (0.8, 1.5),
        }

        # For reasoning model, temperature has no effect per documentation
        if st.session_state["model_select"] == "deepseek-reasoner":
            st.info("⚠️ Temperature has no effect on deepseek-reasoner model")
            st.session_state["temperature"] = 0.0
        else:
            min_temp, max_temp = temp_ranges.get(task_type, (0.3, 0.7))
            st.session_state["temperature"] = st.slider(
                "Temperature",
                min_temp,
                max_temp,
                0.1,
                key="temp_slider",
            )

        # Default prompt section
        default_prompt = st.text_area(
            "Set Default Prompt",
            value=st.session_state["default_prompt"],
            key="default_prompt_area",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Default Prompt"):
                st.session_state["default_prompt"] = default_prompt
                save_default_prompt(default_prompt)
                update_system_message()
                st.success("Default prompt saved!")
        with col2:
            if st.button("Clear Default Prompt"):
                st.session_state["default_prompt"] = ""
                update_system_message()
                st.success("Default prompt cleared!")

# Main chat area
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message DeepSeek..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state["chat_history"].append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        messages = [{"role": "system", "content": st.session_state["system_message"]}]
        messages.extend(st.session_state["chat_history"])

        # Enforce token limits before API call
        messages = enforce_token_limit(messages)
        usage = calculate_context_usage(messages)

        if usage > MAX_API_TOKENS * 0.9:
            st.warning(
                f"High context usage: {usage}/{MAX_API_TOKENS} tokens (API limit)"
            )

        try:
            # UPDATED for V3.1: Handle reasoning model differently
            if st.session_state["model_select"] == "deepseek-reasoner":
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages[-30:],
                    stream=True,
                    max_tokens=MAX_OUTPUT_TOKENS_REASONER,
                )
            else:
                response = client.chat.completions.create(
                    model=st.session_state["model_select"],
                    messages=messages[-30:],
                    stream=True,
                    temperature=st.session_state["temperature"],
                    max_tokens=MAX_OUTPUT_TOKENS_CHAT,
                )

            full_response = ""
            reasoning_content = ""
            container = st.empty()

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    with container.container():
                        with st.chat_message("assistant"):
                            st.markdown(full_response + "▌")

                # NEW: Capture reasoning content for deepseek-reasoner
                if (st.session_state["model_select"] == "deepseek-reasoner" and 
                    chunk.choices and hasattr(chunk.choices[0].delta, 'reasoning_content') and 
                    chunk.choices[0].delta.reasoning_content):
                    reasoning_content += chunk.choices[0].delta.reasoning_content

            # Store reasoning content for deepseek-reasoner
            if st.session_state["model_select"] == "deepseek-reasoner":
                st.session_state["reasoning_content"] = reasoning_content
                if reasoning_content:
                    with st.expander("View Reasoning Process", expanded=False):
                        st.text_area(
                            "Chain of Thought",
                            value=reasoning_content,
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": full_response}
            )

            # NEW: Update cache metrics from API response
            try:
                if hasattr(response, 'usage'):
                    cache_hit = getattr(response.usage, "prompt_cache_hit_tokens", 0)
                    cache_miss = getattr(response.usage, "prompt_cache_miss_tokens", 0)

                    if cache_hit or cache_miss:
                        total = cache_hit + cache_miss
                        efficiency = cache_hit / total if total > 0 else 0

                        st.session_state["cache_metrics"] = {
                            "hit": cache_hit,
                            "miss": cache_miss,
                            "efficiency": efficiency
                        }

                        st.sidebar.info(
                            f"Cache efficiency: {efficiency:.1%}\n"
                            f"Hit: {cache_hit:,} tokens\n"
                            f"Miss: {cache_miss:,} tokens"
                        )
            except Exception as e:
                st.sidebar.warning(f"Could not retrieve cache metrics: {e}")

        except Exception as e:
            st.error(f"API Error: {str(e)}")
            if "rate limit" in str(e).lower():
                st.info("You've hit the rate limit. Please try again in a moment.")