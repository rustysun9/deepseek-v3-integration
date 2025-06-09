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
import re
from typing import List, Dict, Union

# Constants with API-enforced limits
MAX_API_TOKENS = 65536  # DeepSeek API hard limit
CHUNK_SIZE = 32000
MAX_FILE_CONTEXT_LENGTH = 60000  # Reduced to leave room for chat history
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"] 

API_KEY = config("DEEPSEEK_API_KEY", default="")
encoding = tiktoken.encoding_for_model("gpt-4")

# Default context
DEFAULT_CONTEXT = "You are a helpful assistant. Provide clear and concise answers. If you are writing a code make sure to summarize and provide a concise code, optimize the code output to the smartest and the shortest way with better readability and functionality"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
DEFAULT_PROMPT_FILE = "default_prompt.json"

def load_default_prompt():
    """Load default prompt from file if exists, otherwise return empty string"""
    try:
        if os.path.exists(DEFAULT_PROMPT_FILE):
            with open(DEFAULT_PROMPT_FILE, "r") as f:
                return f.read()
    except Exception as e:
        st.error(f"Error loading default prompt: {e}")
    return ""

def save_default_prompt(prompt_text):
    """Save default prompt to file"""
    try:
        with open(DEFAULT_PROMPT_FILE, "w") as f:
            f.write(prompt_text)
    except Exception as e:
        st.error(f"Error saving default prompt: {e}")

def save_chat_session(chat_history):
    """Save current chat session to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CHAT_HISTORY_DIR, f"chat_{timestamp}.json")

        data = {
            "chat_history": chat_history,
            "default_prompt": st.session_state["default_prompt"],
            "file_context": st.session_state["file_context"],
            "system_message": st.session_state["system_message"]
        }

        with open(filename, "w") as f:
            json.dump(data, f)
        return filename
    except Exception as e:
        st.error(f"Error saving chat: {e}")
        return None

def load_chat_session(filename):
    """Load chat session from file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading chat: {e}")
        return None

def list_saved_chats():
    """List all saved chat sessions"""
    return glob.glob(os.path.join(CHAT_HISTORY_DIR, "*.json"))

def enforce_token_limit(messages):
    """Ensure total tokens stay within API limits"""
    total_tokens = calculate_context_usage(messages)

    if total_tokens <= MAX_API_TOKENS:
        return messages

    # Prioritize keeping system message and recent chat history
    system_message = messages[0]
    chat_history = messages[1:]

    # Calculate how many tokens we can allocate to chat history
    system_tokens = calculate_context_usage([system_message])
    remaining_tokens = MAX_API_TOKENS - system_tokens

    # Keep trimming chat history until we're under the limit
    while True:
        chat_history = chat_history[-20:]  # Keep last 20 messages
        history_tokens = calculate_context_usage(chat_history)

        if system_tokens + history_tokens <= MAX_API_TOKENS:
            break

        # If still over, truncate each message content
        for msg in chat_history:
            current_tokens = len(encoding.encode(msg["content"]))
            if current_tokens > 1000:  # Only truncate long messages
                msg["content"] = smart_truncate(msg["content"], current_tokens // 2)

    return [system_message] + chat_history

def smart_truncate(text: str, max_tokens: int, is_code: bool = False) -> str:
    """Optimized truncation for both text and code with structure preservation."""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    if is_code:
        # Code-specific optimized truncation
        return truncate_code(text, max_tokens)

    # Text truncation with improved boundary detection
    return truncate_text(text, max_tokens)

def truncate_code(code: str, max_tokens: int) -> str:
    """Specialized code truncation that preserves structure."""
    # Priority patterns (order matters)
    patterns = [
        (r'^(import|from)\s', 100),      # Imports (high priority)
        (r'^(class|def)\s', 90),         # Class/function definitions
        (r'^@', 80),                     # Decorators
        (r'^#\s*[A-Z]', 70),            # Capitalized comments (likely important)
        (r'^#', 50),                    # Other comments
        (r'.+', 10)                      # Everything else
    ]

    lines = code.split('\n')
    scored_lines = []

    # Score each line based on importance patterns
    for line in lines:
        score = 0
        for pattern, pattern_score in patterns:
            if re.match(pattern, line):
                score = pattern_score
                break
        scored_lines.append((score, line))

    # Sort by score (descending) but keep original order for equal scores
    scored_lines.sort(key=lambda x: (-x[0], lines.index(x[1])))

    # Build output until we hit token limit
    output = []
    current_tokens = 0
    for score, line in scored_lines:
        line_tokens = len(encoding.encode(line))
        if current_tokens + line_tokens > max_tokens:
            continue
        output.append(line)
        current_tokens += line_tokens

    # If we have room, add back some context around high-score lines
    if current_tokens < max_tokens * 0.8:  # If we're using less than 80%
        for line in lines:
            if line not in output:
                line_tokens = len(encoding.encode(line))
                if current_tokens + line_tokens <= max_tokens:
                    output.append(line)
                    current_tokens += line_tokens

    truncated = '\n'.join(output)
    final_tokens = len(encoding.encode(truncated))

    if final_tokens > max_tokens:
        return encoding.decode(encoding.encode(truncated)[:max_tokens])

    if len(output) < len(lines):
        truncated += f"\n\n...[TRUNCATED {len(lines)-len(output)} LINES]..."

    return truncated

def truncate_text(text: str, max_tokens: int) -> str:
    """Optimized text truncation with better boundary detection."""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Try to find a good paragraph break near the middle
    mid_point = max_tokens // 2
    paragraphs = text.split('\n\n')

    if len(paragraphs) > 1:
        # Find the paragraph that contains the mid point
        current_len = 0
        split_index = 0
        for i, para in enumerate(paragraphs):
            para_tokens = len(encoding.encode(para))
            if current_len + para_tokens > mid_point:
                split_index = i
                break
            current_len += para_tokens

        # Build output from first half and last half paragraphs
        first_half = '\n\n'.join(paragraphs[:split_index])
        second_half = '\n\n'.join(paragraphs[split_index:])

        # Calculate how much we can take from each half
        first_tokens = len(encoding.encode(first_half))
        remaining = max(0, max_tokens - first_tokens)

        if remaining > 100:  # Only include second half if we have significant space
            second_tokens = len(encoding.encode(second_half))
            if second_tokens > remaining:
                second_half = encoding.decode(encoding.encode(second_half)[:remaining])

            truncated = f"{first_half}\n\n...[TRUNCATED]...\n\n{second_half}"
        else:
            truncated = first_half
    else:
        # Fallback to simple truncation
        truncated = encoding.decode(tokens[:max_tokens])

    return truncated

def read_large_file(file) -> str:
    """Optimized file reader with better code detection and chunking."""
    # Enhanced code detection
    code_extensions = {
        'py': 'python', 'js': 'javascript', 'java': 'java', 
        'c': 'c', 'cpp': 'cpp', 'h': 'c-header', 
        'sh': 'bash', 'php': 'php', 'rb': 'ruby',
        'go': 'go', 'rs': 'rust', 'ts': 'typescript',
        'md': 'markdown', 'html': 'html', 'css': 'css'
    }

    ext = file.name.split('.')[-1].lower()
    is_code = ext in code_extensions

    try:
        if file.type == "application/pdf":
            doc = fitz.open(stream=io.BytesIO(file.read()), filetype="pdf")
            return "".join(page.get_text() for page in doc)

        # Optimized reading for text/code files
        content = []
        file.seek(0)

        # Read first 1KB to detect encoding
        sample = file.read(1024)
        file.seek(0)
        encoding_info = chardet.detect(sample)
        encoding_type = encoding_info['encoding'] or 'utf-8'

        # Read in optimized chunks (larger for code, smaller for text)
        chunk_size = CHUNK_SIZE * (10 if is_code else 1)

        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            content.append(chunk.decode(encoding_type, errors='replace'))

        full_text = "".join(content)
        return smart_truncate(full_text, MAX_FILE_CONTEXT_LENGTH, is_code)

    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return ""

def get_user_balance():
    """Fetch user balance from DeepSeek API"""
    try:
        response = client._client.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10  # Add timeout
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
        st.progress(min(usage/MAX_API_TOKENS, 1.0))
        st.caption(f"Context usage: {usage:,}/{MAX_API_TOKENS:,} tokens")

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
        with st.spinner(f"Processing {len(uploaded_files)} files..."):
            new_files = []
            for file in uploaded_files:
                with st.expander(f"Processing {file.name}", expanded=False):
                    content = read_large_file(file)
                    if content:
                        new_files.append({"name": file.name, "content": content})
                        st.code(f"Loaded {len(content.splitlines())} lines", language='text')
    
            st.session_state["file_context"] = new_files
            update_system_message()
            
    # Chat management
    if st.session_state["chat_history"]:
        usage = calculate_context_usage(
            [{"role": "system", "content": st.session_state["system_message"]}] +
            st.session_state["chat_history"]
        )
        st.progress(min(usage/MAX_API_TOKENS, 1.0))  # Changed from MAX_TOKENS
        st.caption(f"Context usage: {usage:,}/{MAX_API_TOKENS:,} tokens")

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

        # Enforce token limits before API call
        messages = enforce_token_limit(messages)
        usage = calculate_context_usage(messages)

        if usage > MAX_API_TOKENS * 0.9:
            st.warning(f"High context usage: {usage}/{MAX_API_TOKENS} tokens (API limit)")

        response = client.chat.completions.create(
            model=st.session_state["model_select"],  # Use the selected model
            messages=messages[-30:],
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
                        st.markdown(full_response + "▌")  # Using standard cursor symbol

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