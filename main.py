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
CHAT_NAME_MAX_LENGTH = 50
CHAT_DESC_MAX_LENGTH = 100
MAX_API_TOKENS = 65536  # DeepSeek API hard limit
CHUNK_SIZE = 32000
MAX_FILE_CONTEXT_LENGTH = 60000  # Reduced to leave room for chat history
# Ensure we have an absolute path for chat history
CHAT_HISTORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "chat_history"))
MODELS = ["deepseek-chat", "deepseek-reasoner"]

API_KEY = config("DEEPSEEK_API_KEY", default="")
encoding = tiktoken.encoding_for_model("gpt-4")

# Default context
DEFAULT_CONTEXT = "You are a helpful assistant specializing in code analysis and editing. Provide clear and concise answers. When writing code, ensure it is optimized for readability, functionality, and brevity."

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
            "messages": [],  # Add this if you're using a separate messages list
        }
    )


def save_chat_session(chat_history, chat_name, chat_desc) -> str:
    """Save current chat session with provided name/description"""
    try:
        # Check if input is valid
        if not chat_history or not isinstance(chat_history, list):
            st.sidebar.error(f"Invalid chat history: empty or invalid format")
            return ""
        
        # Generate filename with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        session_id = str(uuid.uuid4())[:8]
        
        # Create full path for the new chat file
        chat_dir_abs = os.path.abspath(CHAT_HISTORY_DIR)
        filename = os.path.join(chat_dir_abs, f"chat_{timestamp}_{session_id}.json")
        temp_path = f"{filename}.tmp"

        # Prepare data structure for saving
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
        }

        # Make sure chat_history directory exists
        os.makedirs(chat_dir_abs, exist_ok=True)
        if not os.path.exists(chat_dir_abs):
            st.sidebar.error(f"Failed to create directory: {chat_dir_abs}")
            return ""
            
        # Write to temporary file first (safer)
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
            
        # Verify temp file was created
        if not os.path.exists(temp_path):
            st.sidebar.error(f"Failed to create temporary file: {temp_path}")
            return ""
            
        # Rename to final filename
        os.rename(temp_path, filename)
        
        # Verify file was renamed successfully
        if os.path.exists(filename):
            # Refresh saved chats list immediately
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
        
        # Validate form submission
        if submitted:
            if not chat_name:
                st.sidebar.error("No chat name provided")
            else:
                return chat_name, chat_desc
    
    # If we get here, either the form wasn't submitted or validation failed
    return None, None


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
            "id": data.get("id", str(uuid.uuid4())),
            "timestamp": data.get(
                "timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
        }


def list_saved_chats():
    """List all saved chat sessions"""
    try:
        # Use absolute path for better reliability
        chat_dir_abs = os.path.abspath(CHAT_HISTORY_DIR)
        
        # Ensure directory exists
        if not os.path.exists(chat_dir_abs):
            os.makedirs(chat_dir_abs, exist_ok=True)
            return []
            
        # Get all JSON files in the directory
        pattern = os.path.join(chat_dir_abs, "*.json")
        
        # Check if directory is accessible
        if not os.access(chat_dir_abs, os.R_OK):
            st.sidebar.error(f"Cannot access chat history directory")
            return []
            
        # Get all files matching pattern
        files = glob.glob(pattern)
        
        # Check if directory exists and is readable
        try:
            # Verify each file is readable and valid
            valid_files = []
            for file in files:
                if os.path.isfile(file) and os.access(file, os.R_OK):
                    try:
                        # Try to read the first few bytes to verify file access
                        with open(file, "rb") as f:
                            f.read(10)
                        valid_files.append(file)
                    except Exception:
                        pass
            
            # Update files to only include valid ones
            files = valid_files
            
        except Exception:
            pass
            
        # Sort by modification time (newest first)
        return sorted(files, key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0, reverse=True)
        
    except Exception as e:
        st.sidebar.error(f"Error listing chats: {str(e)}")
        return []


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
        (r'^\s*#\s*include\s+["<].+[>"]', 100),  # Includes (highest priority)
        (r'^\s*#\s*define\s+\w+', 95),          # Macro definitions
        (r'^\s*(class|struct)\s+\w+', 90),       # Class/struct definitions
        (r'^\s*[\w\s]+\s*\w+\s*\([^)]*\)\s*{', 85),  # Function definitions
        (r'^\s*typedef\s+\w+', 80),             # Typedefs
        (r'^\s*enum\s+\w+', 75),                # Enums
        (r'^\/\*\*?.+?\*\/', 70),               # Documentation comments
        (r'^\/\/.+', 50),                       # Regular comments
        (r'.+', 10)                             # Everything else
    ]

    lines = code.split("\n")
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

    # Try to find a good paragraph break near the middle
    mid_point = max_tokens // 2
    paragraphs = text.split("\n\n")

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
        first_half = "\n\n".join(paragraphs[:split_index])
        second_half = "\n\n".join(paragraphs[split_index:])

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

def get_default_prompt(file_extension):
    prompt_map = {
        '.c': 'Analyze this C code:',
        '.cpp': 'Analyze this C++ code:',
        '.h': 'Analyze this C/C++ header:',
        '.ino': 'Analyze this Arduino sketch:',
        '.py': 'Analyze this Python code:',
        '.java': 'Analyze this Java code:',
        '.js': 'Analyze this JavaScript code:',
        '.ts': 'Analyze this TypeScript code:',
        '.html': 'Analyze this HTML code:',
        '.css': 'Analyze this CSS code:',
        '.rb': 'Analyze this Ruby code:',
        '.php': 'Analyze this PHP code:',
        '.go': 'Analyze this Go code:',
        '.rs': 'Analyze this Rust code:',
        '.swift': 'Analyze this Swift code:',
        '.kt': 'Analyze this Kotlin code:',
        '.sh': 'Analyze this Shell script:',
        '.sql': 'Analyze this SQL script:',
        '.json': 'Analyze this JSON data:',
        '.xml': 'Analyze this XML data:',
        '.yaml': 'Analyze this YAML configuration:',
        '.md': 'Analyze this Markdown document:',
    }
    return prompt_map.get(file_extension, 'Analyze this code:')

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def read_large_file(file) -> str:
    """Optimized file reader with better code detection and chunking."""
    # Enhanced code detection
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
        "ino": "arduino",
        "hpp": "cpp-header",
        "cxx": "cpp",
        "cc": "cpp",
        "hxx": "cpp-header"
    }

    ext = file.name.split(".")[-1].lower()
    is_code = ext in code_extensions

    try:
        if ext in ["ino", "h", "c", "hpp"]:
            # Read header files more aggressively
            chunk_size = CHUNK_SIZE * 20  # Even larger chunks for headers
            is_code = True
    
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
        encoding_type = encoding_info["encoding"] or "utf-8"

        # Read in optimized chunks (larger for code, smaller for text)
        chunk_size = CHUNK_SIZE * (5 if ext in ["ino", "h"] else (10 if is_code else 1))

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
        response = client._client.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,  # Add timeout
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
    """Enhanced for C/C++ projects"""
    system_content = DEFAULT_CONTEXT + "\n\nYou are assisting with a C/C++/Arduino project."

    if st.session_state["file_context"]:
        grouped_files = group_related_files(st.session_state["file_context"])
        file_contexts = []

        # Special handling for .h/.ino pairs
        for group, files in grouped_files.items():
            # Sort files to show headers first
            files.sort(key=lambda x: x['name'].endswith('.h'), reverse=True)

            combined = "\n\n".join([
                f"// File: {f['name']}\n{f['content']}" 
                for f in files
            ])

            # Allocate more tokens to implementation files
            is_implementation = any(f['name'].endswith(('.ino','.c','.cpp')) 
                                for f in files)
            max_tokens = (MAX_FILE_CONTEXT_LENGTH // len(grouped_files)) * (
                2 if is_implementation else 1)

            truncated = smart_truncate(combined, max_tokens, is_code = True)
            file_contexts.append(f"// Project: {group}\n{truncated}")


# Define the group_related_files function at the top level of the file
def group_related_files(file_context):
    """Group related files (e.g., .ino with corresponding .h files) by base name."""
    grouped_files = {}
    for file in file_context:
        base_name = os.path.splitext(file["name"])[0]
        if base_name not in grouped_files:
            grouped_files[base_name] = []
        grouped_files[base_name].append(file)
    return grouped_files


# Function to initialize all session state keys
def init_session_state():
    """Initialize all session state keys with default values"""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "default_prompt" not in st.session_state:
        st.session_state["default_prompt"] = load_default_prompt()

    if "file_context" not in st.session_state:
        st.session_state["file_context"] = []

    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.1  # Default temperature for coding
        
    if "refresh_chats_flag" not in st.session_state:
        st.session_state["refresh_chats_flag"] = True
        
    if "save_chat_clicked" not in st.session_state:
        st.session_state["save_chat_clicked"] = False
        
    if "save_chat_bottom_clicked" not in st.session_state:
        st.session_state["save_chat_bottom_clicked"] = False
        
    if "chat_save_path" not in st.session_state:
        st.session_state["chat_save_path"] = None

    if "system_message" not in st.session_state:
        st.session_state["system_message"] = DEFAULT_CONTEXT
        if st.session_state["default_prompt"]:
            st.session_state["system_message"] += (
                f"\n\nDefault Prompt:\n{st.session_state['default_prompt']}"
            )

# Initialize all session state values
init_session_state()

# Streamlit UI
st.set_page_config(
    initial_sidebar_state="auto", layout="wide", page_title="DeepSeek Chat"
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
        # Hold button state in session to prevent issues with st.button getting reset on interaction
        if "save_chat_clicked" not in st.session_state:
            st.session_state.save_chat_clicked = False
            
        if st.button("💾 Save Chat", key="save_chat_top", help="Save current chat") or st.session_state.save_chat_clicked:
            if not st.session_state["chat_history"]:
                st.error("No chat messages to save!")
            else:
                # Set flag to maintain button "clicked" state during form display
                st.session_state.save_chat_clicked = True
                
                # Make save form more visible
                st.sidebar.markdown("---")
                st.sidebar.markdown("## 📝 Save Your Chat")
                
                # Get chat name and description from modal
                chat_name, chat_desc = save_chat_modal()
                
                if chat_name:
                    # If we got a name, save was confirmed
                    with st.spinner("Saving chat..."):
                        filename = save_chat_session(
                            st.session_state["chat_history"], chat_name, chat_desc
                        )
                        if filename:
                            st.success(f"Saved as {os.path.basename(filename)}")
                            # Reset clicked state
                            st.session_state.save_chat_clicked = False
                            # Force refresh to show new chat in history
                            st.session_state["refresh_chats_flag"] = True
                            st.rerun()

    # Chat history section moved here
    # Initialize refresh flag if not present
    if "refresh_chats_flag" not in st.session_state:
        st.session_state["refresh_chats_flag"] = True
        
    # Get all saved chats with forced refresh
    saved_chats = list_saved_chats()
    
    # Enhanced visibility for chat history section with better styling
    if not saved_chats:
        st.warning("⚠️ No saved chats found. Save a chat to see it here.")
    else:
        # Load metadata for display with error handling
        chat_options = []
        chat_details = {}
        
        for i, chat_file in enumerate(saved_chats):
            try:
                with open(chat_file, "r") as f:
                    data = json.load(f)
                    
                # Extract metadata
                name = data.get("metadata", {}).get("name", os.path.basename(chat_file))
                desc = data.get("metadata", {}).get("description", "")
                created_at = data.get("metadata", {}).get("created_at", "")
                
                # Format date for display if available
                if created_at:
                    try:
                        date_display = datetime.strptime(created_at[:8], "%Y%m%d").strftime("%b %d, %Y")
                    except Exception:
                        date_display = created_at[:8] if len(created_at) >= 8 else created_at
                else:
                    date_display = "Unknown date"
                    
                # Add date to name for better identification
                display_name = f"{name} ({date_display})"
                
                # Get message count
                message_count = len(data.get("chat_history", []))
                
                # Store details
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
        
        # Add a manual refresh button
        if st.button("🔄 Refresh Chat List", key="refresh_chat_list_btn"):
            st.session_state["refresh_chats_flag"] = True
            st.rerun()
        
        if chat_options:
            # Always show saved chats (no expander)
            st.markdown("### Select a saved chat")
            selected_name = st.selectbox(
                "Available chats:", 
                chat_options, 
                key="saved_chat_selector",
                format_func=lambda x: f"{x} ({chat_details[x].get('message_count', 0)} messages)" if x in chat_details else x
            )
            
            if selected_name:
                # Show description and metadata if available
                with st.container():
                    if chat_details[selected_name]["description"]:
                        st.info(chat_details[selected_name]["description"])
                    
                    # Show message count
                    if chat_details[selected_name].get("message_count", 0) > 0:
                        st.caption(f"Contains {chat_details[selected_name]['message_count']} messages")
                    
                    # Show file details
                    st.caption(f"File: {os.path.basename(chat_details[selected_name]['filename'])}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📂 Load Chat", key="load_chat_btn", use_container_width=True):
                            try:
                                with st.spinner(f"Loading {chat_details[selected_name]['original_name']}..."):
                                    data = load_chat_session(chat_details[selected_name]["filename"])
                                    st.session_state.update(
                                        {
                                            "chat_history": data["chat_history"],
                                            "default_prompt": data["default_prompt"],
                                            "file_context": data["file_context"],
                                            "system_message": data["system_message"],
                                        }
                                    )
                                    st.success(f"Loaded: {chat_details[selected_name]['original_name']}")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Failed to load chat: {e}")
                    with col2:
                        if st.button("🗑️ Delete", key="delete_chat_btn", use_container_width=True):
                            try:
                                full_path = chat_details[selected_name]["filename"]
                                os.remove(full_path)
                                st.success(f"Deleted {chat_details[selected_name]['original_name']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting file: {e}")

    # File upload section moved here
    st.sidebar.markdown("## Files Management")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=[
            "pdf",
            "txt",
            "py",
            "md",
            "csv",
            "json",
            "docx",
            "h",
            "ino",
            "sh",
            "php",
            "js",
            "html",
            "cmd",
            "map",
            "c",
        ],
        accept_multiple_files=True,
    )

    def generate_dynamic_default_prompt(uploaded_files):
        """Generate a dynamic default prompt based on uploaded files."""
        # Determine project type based on file extensions
        file_extensions = set(
            get_file_extension(file["name"] if isinstance(file, dict) else file.name)
            for file in uploaded_files
        )
        project_type_map = {
            '.ino': 'arduino',
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'c++',
            '.h': 'c/c++ header',
            '.sh': 'shell script',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
        }

        # Determine the most relevant project type
        project_type = "general"
        for ext, type_name in project_type_map.items():
            if ext in file_extensions:
                project_type = type_name
                break

        # Generate the dynamic prompt
        extensions_list = ", ".join(sorted(file_extensions))
        return f"This is a \"{project_type}\" project. Uploaded project files are {extensions_list} files."

    # Update the dynamic default prompt generation in the file upload section
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

            # Generate and set the dynamic default prompt
            dynamic_prompt = generate_dynamic_default_prompt(uploaded_files)
            st.session_state["default_prompt"] = dynamic_prompt
            update_system_message()

            st.success("Dynamic default prompt generated and applied!")

    if st.session_state.get("file_context"):
        if st.button(
            "🗑️ Delete All Files",
            key="delete_all_files",
            help="Remove all uploaded files at once",
        ):
            st.session_state["file_context"] = []  # Clear all files
            st.session_state["default_prompt"] = ""  # Clear the default prompt
            update_system_message()  # Update context
            st.success("All uploaded files have been removed!")
            st.session_state.pop("uploaded_files", None)  # Reset file uploader
            st.query_params.from_dict({})  # Clear query parameters to refresh the app

    st.title("Settings")

    # Context usage display at top
    if st.session_state["chat_history"]:
        usage = calculate_context_usage(
            [{"role": "system", "content": st.session_state["system_message"]}]
            + st.session_state["chat_history"]
        )
        st.progress(min(usage / MAX_API_TOKENS, 1.0))
        st.caption(f"Context usage: {usage:,}/{MAX_API_TOKENS:,} tokens")

    # Account Balance Section
    st.subheader("Account Balance")
    if st.button("🔄 Refresh Balance"):
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

    # Advanced Settings moved to bottom of sidebar
    with st.expander("Advanced Settings"):
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

        # Model selection
        model_choice = st.selectbox("Model", MODELS, index=0, key="model_select")

        # Task selection
        task_type = st.selectbox(
            "Task Type",
            ["Coding/Math", "Normal Questions", "Data Analysis", "Creative Writing"],
            index=0,  # Default to Coding/Math
            key="task_type_select",
        )

        # Dynamic temperature
        temp_ranges = {
            "Coding/Math": (0.0, 0.3),
            "Data Analysis": (0.3, 0.7),
            "Normal Questions": (0.5, 0.9),
            "Creative Writing": (0.8, 1.5),
        }
        min_temp, max_temp = temp_ranges.get(task_type, (0.3, 0.7))
        st.session_state["temperature"] = st.slider(
            "Temperature",
            min_temp,
            max_temp,
            0.1,  # Default temperature for coding
            key="temp_slider",
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
            st.warning(
                f"High context usage: {usage}/{MAX_API_TOKENS} tokens (API limit)"
            )

        response = client.chat.completions.create(
            model=st.session_state["model_select"],  # Use the selected model
            messages=messages[-30:],
            stream=True,
            temperature=st.session_state["temperature"],
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
                        f"Cache efficiency: {cache_hit / (cache_hit + cache_miss):.1%}\n"
                        f"Hit: {cache_hit}, Miss: {cache_miss}"
                    )
        except Exception:
            pass

# Remove the manual filename input and ensure dynamic prompt generation is used
if __name__ == "__main__":
    # Ensure the dynamic prompt is generated and applied if files are uploaded
    if "file_context" in st.session_state and st.session_state["file_context"]:
        dynamic_prompt = generate_dynamic_default_prompt(
            [file for file in st.session_state["file_context"]]
        )
        st.session_state["default_prompt"] = dynamic_prompt
        update_system_message()
        print(f"Dynamic default prompt applied: {dynamic_prompt}")
    else:
        print("No files uploaded. Default prompt not generated.")