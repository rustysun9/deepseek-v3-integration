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
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

MAX_TOKENS = 65000
encoding = tiktoken.encoding_for_model("gpt-4")
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

API_KEY = config("DEEPSEEK_API_KEY", default="")

# Default context (first priority)
DEFAULT_CONTEXT = """You are an expert programming assistant. When working with code:
1. Analyze cross-file dependencies when answering questions
2. Pay attention to imports/includes between files
3. Provide answers that consider the entire codebase context
4. For code suggestions, maintain consistency with existing code style
5. When showing code, indicate which file it belongs to"""

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

DEFAULT_PROMPT_FILE = "default_prompt.json"
MAX_FILE_CONTEXT_LENGTH = 65000

# Supported code file types and their languages
CODE_FILE_TYPES = {
    "py": "python",
    "js": "javascript",
    "java": "java",
    "cpp": "cpp",
    "h": "cpp",
    "hpp": "cpp",
    "ino": "arduino",
    "md": "markdown",
    "sh": "bash",
    "txt": "text",
}


def get_file_language(filename):
    ext = filename.split(".")[-1].lower()
    return CODE_FILE_TYPES.get(ext, "text")


def syntax_highlight(code, language):
    try:
        lexer = get_lexer_by_name(language)
        formatter = HtmlFormatter(style="colorful")
        return highlight(code, lexer, formatter)
    except:
        return f"<pre><code>{code}</code></pre>"


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
                "focused_files": st.session_state.get("focused_files", []),
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


def truncate_to_token_limit(text, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = encoding.decode(tokens[:max_tokens])
    return truncated + "\n...[TRUNCATED]"


def analyze_dependencies(content, language):
    """Analyze code dependencies based on language"""
    dependencies = []

    if language == "python":
        imports = re.findall(r"^import\s+(\w+)", content, re.M)
        imports += re.findall(r"^from\s+(\w+)", content, re.M)
        dependencies = list(set(imports))
    elif language in ["cpp", "arduino"]:
        includes = re.findall(r'^#include\s+[<"]([\w\./]+)[>"]', content, re.M)
        dependencies = list(set(includes))
    elif language == "javascript":
        requires = re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', content)
        imports = re.findall(r'import\s+.+\s+from\s+[\'"]([^\'"]+)[\'"]', content)
        dependencies = list(set(requires + imports))

    return dependencies


def clean_code_content(content, language):
    """Basic code cleaning while preserving important structure"""
    # Remove excessive blank lines
    content = re.sub(r"\n\s*\n", "\n\n", content)

    # For non-code files, just return as-is
    if language == "text":
        return content

    return content


def read_pdf(file):
    try:
        pdf_bytes = file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def read_file(file):
    try:
        if file.type == "application/pdf":
            return read_pdf(file)
        else:
            raw_data = file.read()
            encoding_info = chardet.detect(raw_data)
            encoding = (
                encoding_info["encoding"] if encoding_info["encoding"] else "utf-8"
            )
            return raw_data.decode(encoding, errors="replace")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "default_prompt" not in st.session_state:
    st.session_state["default_prompt"] = load_default_prompt()

if "file_context" not in st.session_state:
    st.session_state["file_context"] = []

if "file_cache" not in st.session_state:
    st.session_state["file_cache"] = {}

if "focused_files" not in st.session_state:
    st.session_state["focused_files"] = []

if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0

if "system_message" not in st.session_state:
    st.session_state["system_message"] = DEFAULT_CONTEXT
    if st.session_state["default_prompt"]:
        st.session_state["system_message"] += (
            f"\n\nDefault Prompt:\n{st.session_state['default_prompt']}"
        )


def update_system_message():
    """Enhanced system message with code context awareness"""
    system_content = DEFAULT_CONTEXT

    if st.session_state["default_prompt"]:
        system_content += f"\n\nDefault Prompt:\n{st.session_state['default_prompt']}"

    if st.session_state["file_context"]:
        # Prepare file information
        files_info = []
        for file in st.session_state["file_context"]:
            language = get_file_language(file["name"])
            cleaned_content = clean_code_content(file["content"], language)
            files_info.append(
                {
                    "name": file["name"],
                    "language": language,
                    "content": cleaned_content,
                    "size": len(cleaned_content),
                    "is_focused": file["name"] in st.session_state["focused_files"],
                }
            )

        # Analyze dependencies between files
        file_relationships = []
        for file in files_info:
            dependencies = analyze_dependencies(file["content"], file["language"])
            file_relationships.append(
                {"name": file["name"], "dependencies": dependencies, "depended_by": []}
            )

        # Build dependency graph
        for file in file_relationships:
            for dep in file["dependencies"]:
                for target in file_relationships:
                    if dep in target["name"]:
                        target["depended_by"].append(file["name"])

        # Build context with awareness of relationships and focus
        all_file_contexts = []
        for file in files_info:
            # Find relationship info
            rel_info = next(
                (f for f in file_relationships if f["name"] == file["name"]), None
            )

            file_header = (
                f"File: {file['name']} ({file['language']}, {file['size']} chars)"
            )

            if rel_info:
                if rel_info["dependencies"]:
                    file_header += (
                        f"\nDepends on: {', '.join(rel_info['dependencies'])}"
                    )
                if rel_info["depended_by"]:
                    file_header += (
                        f"\nRequired by: {', '.join(rel_info['depended_by'])}"
                    )

            if file["is_focused"]:
                file_header += "\n⭐ CURRENTLY FOCUSED"

            file_summary = f"{file_header}\nContent:\n{file['content']}"
            all_file_contexts.append(file_summary)

        joined_file_context = "\n\n".join(all_file_contexts)
        joined_file_context = truncate_to_token_limit(joined_file_context, MAX_TOKENS)

        system_content += f"\n\nCodebase Context:\n{joined_file_context}"

    st.session_state["system_message"] = system_content


def call_deepseek_api(messages, streaming=True, temperature=None):
    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            stream=streaming,
            temperature=temperature,
        )
        return response
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return None


# Custom CSS for better code display
st.markdown(
    """
    <style>
        .fixed-bottom {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem;
            background: white;
            z-index: 100;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        
        .main-container {
            padding-bottom: 120px;
        }
        
        .auto-scroll {
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }
        
        .code-block {
            background-color: #f8f8f8;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            overflow-x: auto;
        }
        
        .file-navigator {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        
        .focused-file {
            background-color: #e6f7ff;
            border-left: 3px solid #1890ff;
            padding-left: 10px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("Code-Aware AI Assistant (DeepSeek)")

# Sidebar settings
with st.sidebar:
    st.title("Settings")

    # Default prompt section
    default_prompt = st.text_area(
        "Set Default Prompt", value=st.session_state["default_prompt"]
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Default Prompt"):
            st.session_state["default_prompt"] = default_prompt
            save_default_prompt(default_prompt)
            update_system_message()
    with col2:
        if st.button("Clear Default Prompt"):
            st.session_state["default_prompt"] = ""
            update_system_message()

    # Task type selection
    task_type = st.selectbox(
        "Select Task Type",
        [
            "Code Analysis",
            "Debugging",
            "Feature Implementation",
            "Code Review",
            "Documentation",
            "General Questions",
        ],
        index=0,
    )

    # Set temperature based on task type
    if task_type in ["Code Analysis", "Debugging"]:
        temperature = 0.1
    elif task_type == "Feature Implementation":
        temperature = 0.3
    elif task_type == "Code Review":
        temperature = 0.5
    elif task_type == "Documentation":
        temperature = 0.7
    else:
        temperature = 0.9

    st.write(f"Selected Temperature: {temperature}")

    # Model selection
    models = ["deepseek-chat", "deepseek-reasoner"]
    model_choice = st.selectbox("Choose a model", models, index=0)

    # Always show the file uploader, regardless of clearing_files state
    if "clearing_files" not in st.session_state:
        st.session_state.clearing_files = False

    if not st.session_state.clearing_files:
        uploaded_files = st.file_uploader(
            "Upload code files",
            type=list(CODE_FILE_TYPES.keys()),
            accept_multiple_files=True,
        )
    else:
        uploaded_files = []
        st.session_state.clearing_files = False

    if uploaded_files:
        new_files = []
        for uploaded_file in uploaded_files:
            # Check if file is already in file_context (not just in cache)
            if not any(
                f["name"] == uploaded_file.name
                for f in st.session_state["file_context"]
            ):
                if uploaded_file.name not in st.session_state.file_cache:
                    file_content = read_file(uploaded_file)
                    if file_content:
                        st.session_state.file_cache[uploaded_file.name] = file_content
                        new_files.append(
                            {"name": uploaded_file.name, "content": file_content}
                        )
                else:
                    new_files.append(
                        {
                            "name": uploaded_file.name,
                            "content": st.session_state.file_cache[uploaded_file.name],
                        }
                    )

        if new_files:
            st.session_state["file_context"].extend(new_files)
            update_system_message()
            st.success(f"Added {len(new_files)} file(s)")

    # File navigator
    if st.session_state["file_context"]:
        st.subheader("File Navigator")
        
        with st.container():
            st.markdown('<div class="file-navigator">', unsafe_allow_html=True)
            
            for i, file in enumerate(st.session_state["file_context"]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"📄 {file['name']}", key=f"show_{file['name']}_{i}"):
                        st.session_state["focused_files"] = [file['name']]
                        update_system_message()
                with col2:
                    if st.button("❌", key=f"remove_{file['name']}_{i}"):
                        st.session_state.clearing_files = True
                        st.session_state["file_context"] = [
                            f for f in st.session_state["file_context"] 
                            if f['name'] != file['name']
                        ]
                        if file['name'] in st.session_state["focused_files"]:
                            st.session_state["focused_files"].remove(file['name'])
                        update_system_message()
                        st.rerun()
            
            # Moved outside the loop and added a unique key
            if st.button("Clear All Files", key="clear_all_files_button"):
                st.session_state.clearing_files = True
                st.session_state["file_context"] = []
                st.session_state["focused_files"] = []
                st.session_state.file_cache = {}
                update_system_message()
                st.rerun()

    # Focus management
    if st.session_state["file_context"] and len(st.session_state["file_context"]) > 1:
        focus_options = [f["name"] for f in st.session_state["file_context"]]
        selected_focus = st.multiselect(
            "Focus on specific files (optional)",
            focus_options,
            default=st.session_state["focused_files"],
        )

        if set(selected_focus) != set(st.session_state["focused_files"]):
            st.session_state["focused_files"] = selected_focus
            update_system_message()

    # Chat management
    st.subheader("Chat History")

    if st.button("💾 Save Current Chat"):
        if st.session_state["chat_history"]:
            filename = save_chat_session(st.session_state["chat_history"])
            st.success(f"Chat saved as {os.path.basename(filename)}")
        else:
            st.warning("No chat history to save")

    saved_chats = list_saved_chats()
    if saved_chats:
        selected_chat = st.selectbox(
            "Select a chat to load",
            [os.path.basename(f) for f in saved_chats],
            index=0,
            key="chat_selector",
        )

        if st.button("📂 Load Selected Chat"):
            selected_file = os.path.join(CHAT_HISTORY_DIR, selected_chat)
            chat_data = load_chat_session(selected_file)

            st.session_state.update(
                {
                    "chat_history": chat_data["chat_history"],
                    "default_prompt": chat_data["default_prompt"],
                    "file_context": chat_data["file_context"],
                    "system_message": chat_data.get("system_message", DEFAULT_CONTEXT),
                    "focused_files": chat_data.get("focused_files", []),
                }
            )
            st.success(f"Loaded chat: {selected_chat}")
            st.rerun()

        if st.button("🗑️ Delete Selected Chat"):
            selected_file = os.path.join(CHAT_HISTORY_DIR, selected_chat)
            os.remove(selected_file)
            st.success(f"Deleted chat: {selected_chat}")
            st.rerun()
    else:
        st.write("No saved chats yet")

# Main chat area
with st.container():
    st.markdown('<div class="main-container auto-scroll">', unsafe_allow_html=True)

    for message in st.session_state["chat_history"]:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            # Check if content contains code blocks and apply syntax highlighting
            if "```" in content:
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd parts are code blocks
                        # Try to detect language from the code block
                        language = part.split("\n")[0].strip() or "text"
                        code = "\n".join(part.split("\n")[1:])
                        st.markdown(
                            syntax_highlight(code, language), unsafe_allow_html=True
                        )
                    else:
                        st.markdown(part)
            else:
                st.markdown(content)

    if "temp_response" in st.session_state:
        with st.chat_message("assistant"):
            st.markdown(st.session_state["temp_response"] + "▌")

    st.markdown("</div>", unsafe_allow_html=True)

# Fixed input at bottom
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
user_input = st.chat_input("Ask about your code...")
st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll JavaScript
auto_scroll_js = """
<script>
function scrollToBottom() {
    window.parent.document.querySelector('.auto-scroll').scrollTop = window.parent.document.querySelector('.auto-scroll').scrollHeight;
}
scrollToBottom();
const observer = new MutationObserver(scrollToBottom);
observer.observe(window.parent.document.querySelector('.auto-scroll'), {
    childList: true,
    subtree: true
});
</script>
"""
st.components.v1.html(auto_scroll_js, height=0)

if user_input:
    with st.spinner("Analyzing code..."):
        messages = [{"role": "system", "content": st.session_state["system_message"]}]
        messages.extend(st.session_state["chat_history"])
        messages.append({"role": "user", "content": user_input})

        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.container():
            with st.chat_message("user"):
                st.markdown(user_input)

        response = call_deepseek_api(messages=messages, temperature=temperature)

        if response:
            full_response = ""
            response_placeholder = st.empty()

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    st.session_state["temp_response"] = full_response
                    with response_placeholder.container():
                        with st.chat_message("assistant"):
                            st.markdown(full_response + "▌")

            del st.session_state["temp_response"]
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": full_response}
            )

            try:
                if hasattr(response, "usage") and hasattr(
                    response.usage, "prompt_cache_hit_tokens"
                ):
                    st.sidebar.info(
                        f"Cache hit tokens: {response.usage.prompt_cache_hit_tokens}, Cache miss tokens: {response.usage.prompt_cache_miss_tokens}"
                    )
            except Exception as e:
                st.sidebar.warning(f"Could not retrieve cache info: {str(e)}")

            st.rerun()
        else:
            st.warning("No response received from the API.")

