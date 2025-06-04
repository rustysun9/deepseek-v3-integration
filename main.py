import io
import chardet
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

# Constants with 128K context support
MAX_TOKENS = 128000
CHUNK_SIZE = 32000
MAX_FILE_CONTEXT_LENGTH = 120000
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"] 

API_KEY = config("DEEPSEEK_API_KEY", default="")
encoding = tiktoken.encoding_for_model("gpt-4")

# Default context
DEFAULT_CONTEXT = "You are a helpful assistant. Provide clear and concise answers. If you are writing a code make sure to summarize and provide a concise code, optimize the code output to the smartest and the shortest way with better readability and functionality"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
DEFAULT_PROMPT_FILE = "default_prompt.json"

def save_chat_session(chat_history):
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{CHAT_HISTORY_DIR}/{timestamp}_{session_id}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "id": session_id,
                "timestamp": timestamp,
                "chat_history": chat_history,
                "default_prompt": st.session_state.get("default_prompt", ""),
                "file_context": st.session_state.get("file_context", []),
                "system_message": st.session_state.get("system_message", ""),
            },
            f,
        )
    return filename

def load_chat_session(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data

def list_saved_chats():
    files = sorted(glob.glob(f"{CHAT_HISTORY_DIR}/*.json"), reverse=True)
    return files

def save_default_prompt(prompt):
    with open(DEFAULT_PROMPT_FILE, "w") as f:
        json.dump({"prompt": prompt}, f)

def load_default_prompt():
    if os.path.exists(DEFAULT_PROMPT_FILE):
        with open(DEFAULT_PROMPT_FILE, "r") as f:
            return json.load(f).get("prompt", "")
    return ""

def smart_truncate(text, max_tokens):
    """Smart truncation that preserves important sections"""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Keep beginning and end, with middle truncated
    keep_start = tokens[:max_tokens//3]
    keep_end = tokens[-(max_tokens//3):]
    truncated = encoding.decode(keep_start + keep_end)
    return f"{truncated}\n...[TRUNCATED {len(tokens)-max_tokens} TOKENS]..."

def read_large_file(file):
    """Read large files in chunks"""
    try:
        if file.type == "application/pdf":
            doc = fitz.open(stream=io.BytesIO(file.read()), filetype="pdf")
            return "".join(page.get_text() for page in doc)
        else:
            content = []
            file.seek(0)
            while True:
                chunk = file.read(CHUNK_SIZE)
                if not chunk:
                    break
                encoding_info = chardet.detect(chunk)
                content.append(chunk.decode(encoding_info['encoding'], errors='replace'))
            return "".join(content)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def get_user_balance():
    """Fetch user balance from DeepSeek API"""
    try:
        response = client._client.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("is_available", False):
            st.sidebar.warning("Balance information not available")
            return None

        return data.get("balance_infos", [{}])[0]  # Return first balance info
    except Exception as e:
        st.sidebar.error(f"Failed to fetch balance: {e}")
        return None

# Initialize balance on first load
if "balance_info" not in st.session_state:
    st.session_state.balance_info = get_user_balance()

def calculate_context_usage(messages):
    """Calculate total token usage"""
    return sum(len(encoding.encode(msg["content"])) for msg in messages)

def update_system_message():
    """Update system message with current context"""
    system_content = DEFAULT_CONTEXT

    if st.session_state["default_prompt"]:
        system_content += f"\n\nDefault Prompt:\n{st.session_state['default_prompt']}"

    if st.session_state["file_context"]:
        file_contexts = []
        for file in st.session_state["file_context"]:
            truncated = smart_truncate(file['content'], MAX_FILE_CONTEXT_LENGTH//len(st.session_state["file_context"]))
            file_contexts.append(f"File: {file['name']}\nContent:\n{truncated}")

        system_content += f"\n\nUploaded Files Context:\n" + "\n\n".join(file_contexts)

    st.session_state["system_message"] = smart_truncate(system_content, MAX_FILE_CONTEXT_LENGTH)

# Initialize session state with default values
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "default_prompt" not in st.session_state:
    st.session_state["default_prompt"] = load_default_prompt()

if "file_context" not in st.session_state:
    st.session_state["file_context"] = []

if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.1  # Default temperature for coding

if "system_message" not in st.session_state:
    st.session_state["system_message"] = DEFAULT_CONTEXT
    if st.session_state["default_prompt"]:
        st.session_state["system_message"] += f"\n\nDefault Prompt:\n{st.session_state['default_prompt']}"

# Streamlit UI
st.set_page_config(layout="wide")
st.title("DeepSeek Chat (128K Context Support)")

# Sidebar
with st.sidebar:
    st.title("Settings")

    # Context usage display at top
    if st.session_state["chat_history"]:
        usage = calculate_context_usage(
            [{"role": "system", "content": st.session_state["system_message"]}] +
            st.session_state["chat_history"]
        )
        st.progress(min(usage/MAX_TOKENS, 1.0))
        st.caption(f"Context usage: {usage:,}/{MAX_TOKENS:,} tokens")

    st.subheader("Account Balance")
    if st.button("🔄 Refresh Balance"):
        st.session_state.balance_info = get_user_balance()

    if st.session_state.balance_info:
        balance = st.session_state.balance_info
        st.metric("Total Balance", f"{balance.get('total_balance', 0)} {balance.get('currency', 'USD')}")
        st.metric("Granted", f"{balance.get('granted_balance', 0)} {balance.get('currency', 'USD')}")
        st.metric("Topped Up", f"{balance.get('topped_up_balance', 0)} {balance.get('currency', 'USD')}")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Context size control
    current_max_tokens = st.slider(
        "Max Context Size (in tokens)",
        8000, 128000, 128000,
        help="Larger values remember more but may be slower"
    )

    # File upload
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "py", "md", "csv", "json", "docx", "h", "ino", "sh", "php", "js", "html", "cmd", "map", "map"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = []
        for file in uploaded_files:
            content = read_large_file(file)
            if content:
                new_files.append({"name": file.name, "content": content})

        st.session_state["file_context"] = new_files
        update_system_message()
        st.success(f"Loaded {len(new_files)} file(s)")

    # Chat management
    if st.button("🧩 New Chat"):
        if st.session_state["chat_history"]:
            save_chat_session(st.session_state["chat_history"])
        st.session_state["chat_history"] = []
        st.rerun()

    if st.button("💾 Save Current Chat"):
        if st.session_state["chat_history"]:
            filename = save_chat_session(st.session_state["chat_history"])
            st.success(f"Saved as {os.path.basename(filename)}")

    # Chat history
    st.subheader("Chat History")
    saved_chats = list_saved_chats()
    if saved_chats:
        selected = st.selectbox(
            "Saved chats", 
            [os.path.basename(f) for f in saved_chats]
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Load"):
                data = load_chat_session(os.path.join(CHAT_HISTORY_DIR, selected))
                st.session_state.update({
                    "chat_history": data["chat_history"],
                    "default_prompt": data["default_prompt"],
                    "file_context": data["file_context"],
                    "system_message": data["system_message"]
                })
                st.rerun()
        with col2:
            if st.button("❌ Delete"):
                os.remove(os.path.join(CHAT_HISTORY_DIR, selected))
                st.rerun()

    # Advanced Settings moved to bottom of sidebar
    with st.expander("Advanced Settings"):
        # Default prompt section
        default_prompt = st.text_area(
            "Set Default Prompt", 
            value=st.session_state["default_prompt"],
            key="default_prompt_area"
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

        # Model selection
        model_choice = st.selectbox(
            "Model",
            MODELS,
            index=0,
            key="model_select"
        )

        # Task selection
        task_type = st.selectbox(
            "Task Type",
            ["Coding/Math", "Normal Questions", "Data Analysis", "Creative Writing"],
            index=0,  # Default to Coding/Math
            key="task_type_select"
        )

        # Dynamic temperature
        temp_ranges = {
            "Coding/Math": (0.0, 0.3),
            "Data Analysis": (0.3, 0.7),
            "Normal Questions": (0.5, 0.9),
            "Creative Writing": (0.8, 1.5)
        }
        min_temp, max_temp = temp_ranges.get(task_type, (0.3, 0.7))
        st.session_state["temperature"] = st.slider(
            "Temperature", 
            min_temp, 
            max_temp, 
            0.1,  # Default temperature for coding
            key="temp_slider"
        )

# Main chat area (keep this the same as before)
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

        # Context usage indicator
        usage = calculate_context_usage(messages)
        if usage > current_max_tokens * 0.9:
            st.warning(f"High context usage: {usage}/{current_max_tokens} tokens")

        response = client.chat.completions.create(
            model=model_choice,
            messages=messages[-50:],  # Limit to last 50 messages
            stream=True,
            temperature=st.session_state["temperature"]
        )

        full_response = ""
        container = st.empty()
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                with container.container():
                    with st.chat_message("assistant"):
                        st.markdown(full_response + "▌")

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": full_response}
        )

        # Show cache info if available
        try:
            if hasattr(response, "usage"):
                cache_hit = getattr(response.usage, "prompt_cache_hit_tokens", 0)
                cache_miss = getattr(response.usage, "prompt_cache_miss_tokens", 0)
                if cache_hit or cache_miss:
                    st.sidebar.info(
                        f"Cache efficiency: {cache_hit/(cache_hit+cache_miss):.1%}\n"
                        f"Hit: {cache_hit}, Miss: {cache_miss}"
                    )
        except Exception:
            pass