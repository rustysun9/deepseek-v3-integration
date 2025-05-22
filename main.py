import io
import chardet
import fitz
import streamlit as st
from decouple import config
from openai import OpenAI
import json
import os
import textwrap

API_KEY = config("DEEPSEEK_API_KEY", default="")

# Default context (first priority)
DEFAULT_CONTEXT = "You are a helpful assistant. Provide clear and concise answers. If you are writing a code make sure to summarize and provide a concise code, optimize the code output to the smartest and the shortest way with better readability and functionality"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

DEFAULT_PROMPT_FILE = "default_prompt.json"
MAX_FILE_CONTEXT_LENGTH = 50000  # Leave room for conversation history
CHUNK_SIZE = 10000  # For processing large files

def save_default_prompt(prompt):
    with open(DEFAULT_PROMPT_FILE, 'w') as f:
        json.dump({"prompt": prompt}, f)

def load_default_prompt():
    if os.path.exists(DEFAULT_PROMPT_FILE):
        with open(DEFAULT_PROMPT_FILE, 'r') as f:
            return json.load(f).get("prompt", "")
    return ""

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "file_context" not in st.session_state:
    st.session_state["file_context"] = ""
if "default_prompt" not in st.session_state:
    st.session_state["default_prompt"] = load_default_prompt()
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0
if "current_file_index" not in st.session_state:
    st.session_state["current_file_index"] = 0

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

def process_large_text(text, max_length):
    """Process large text by splitting into chunks if needed"""
    if len(text) <= max_length:
        return text
    
    # Try to split at natural boundaries first
    chunks = textwrap.wrap(text, width=max_length, break_long_words=False)
    if len(chunks) == 1:
        # If still too large, split by lines
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) + 1 > max_length and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(line)
            current_length += len(line) + 1
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
    
    return chunks[0]  # Return first chunk for initial context

def read_file(file):
    try:
        if file.type == "application/pdf":
            return read_pdf(file)
        else:
            raw_data = file.read()
            encoding_info = chardet.detect(raw_data)
            encoding = encoding_info["encoding"]
            return raw_data.decode(encoding, errors="replace")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# Custom CSS to fix input at bottom and control scrolling
st.markdown("""
    <style>
        /* Fix the input container at the bottom */
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
        
        /* Add padding to the main container to prevent content hiding */
        .main-container {
            padding-bottom: 120px;
        }
        
        /* Auto-scrolling for chat */
        .auto-scroll {
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }
        
        /* File navigation buttons */
        .file-nav {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("My Own GPT powering DeepSeek model")

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
    with col2:
        if st.button("Clear Default Prompt"):
            st.session_state["default_prompt"] = ""
    
    # Task type selection
    task_type = st.selectbox(
        "Select Task Type",
        [
            "Normal Questions",
            "Coding/Math Questions",
            "Computational Tasks",
            "Creative Tasks/Poetry",
            "File Questions",
            "Data Cleaning/Data Analysis",
            "General Conversation",
            "Translation",
        ],
        index=1,
    )
    
    # Set temperature based on task type
    if task_type == "Coding/Math Questions":
        temperature = 0.0
    elif task_type == "Normal Questions":
        temperature = 0.6
    elif task_type == "File Questions":
        temperature = 0.1
    elif task_type == "Computational Tasks":
        temperature = 0.2
    elif task_type == "Data Cleaning/Data Analysis":
        temperature = 1.0
    elif task_type in ["General Conversation", "Translation"]:
        temperature = 1.3
    elif task_type == "Creative Tasks/Poetry":
        temperature = 1.5
    
    st.write(f"Selected Temperature: {temperature}")
    
    # Model selection
    models = ["deepseek-chat", "deepseek-reasoner"]
    model_choice = st.selectbox("Choose a model", models, index=0)
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload files for context",
        type=["pdf", "docx", "txt", "ino", "h", "py", "md", "sh"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        all_file_contents = []
        for uploaded_file in uploaded_files:
            file_content = read_file(uploaded_file)
            if file_content:
                processed_content = process_large_text(file_content, MAX_FILE_CONTEXT_LENGTH)
                all_file_contents.append({
                    "name": uploaded_file.name,
                    "content": processed_content,
                    "full_content": file_content
                })

        st.session_state["file_context"] = all_file_contents
        st.session_state["current_file_index"] = 0

        st.success(f"Loaded {len(uploaded_files)} file(s)")
        if st.checkbox("Show file contents"):
            for idx, file in enumerate(all_file_contents):
                st.subheader(file['name'])
                st.text_area(f"Content - {file['name']}", file['content'], height=200)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()

# Main chat area container with auto-scrolling
with st.container():
    st.markdown('<div class="main-container auto-scroll">', unsafe_allow_html=True)
    
    # Display chat history in chronological order (newest at bottom)
    for message in st.session_state["chat_history"]:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.markdown(content)
    
    # Display temporary streaming response if exists
    if "temp_response" in st.session_state:
        with st.chat_message("assistant"):
            st.markdown(st.session_state["temp_response"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# File navigation if multiple files are uploaded
if "file_context" in st.session_state and len(st.session_state["file_context"]) > 1:
    st.markdown('<div class="file-nav">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous File"):
            st.session_state["current_file_index"] = max(0, st.session_state["current_file_index"] - 1)
            st.rerun()
    with col2:
        if st.button("Next File"):
            st.session_state["current_file_index"] = min(
                len(st.session_state["file_context"]) - 1,
                st.session_state["current_file_index"] + 1
            )
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    current_file = st.session_state["file_context"][st.session_state["current_file_index"]]
    st.info(f"Currently viewing: {current_file['name']}")

# Fixed input at bottom
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
user_input = st.chat_input("Enter your message here...")
st.markdown('</div>', unsafe_allow_html=True)

# JavaScript for auto-scrolling
auto_scroll_js = """
<script>
function scrollToBottom() {
    window.parent.document.querySelector('.auto-scroll').scrollTop = window.parent.document.querySelector('.auto-scroll').scrollHeight;
}
// Scroll initially
scrollToBottom();
// Set up MutationObserver to detect new messages
const observer = new MutationObserver(scrollToBottom);
observer.observe(window.parent.document.querySelector('.auto-scroll'), {
    childList: true,
    subtree: true
});
</script>
"""
st.components.v1.html(auto_scroll_js, height=0)

if user_input:
    with st.spinner("Thinking..."):
        # Prepare messages with context
        system_content = DEFAULT_CONTEXT
        if st.session_state["default_prompt"]:
            system_content += f"\n\nDefault Prompt:\n{st.session_state['default_prompt']}"
        
        if "file_context" in st.session_state and st.session_state["file_context"]:
            current_file = st.session_state["file_context"][st.session_state["current_file_index"]]
            file_context = f"File: {current_file['name']}\nContent:\n{current_file['content']}"
            system_content += f"\n\nCurrent File Context:\n{file_context}"

        # Build message history - keep only recent messages to save context space
        messages = [{"role": "system", "content": system_content}]
        
        # Add only the most recent messages to conserve context length
        max_history_messages = 5  # Adjust based on your needs
        recent_history = st.session_state["chat_history"][-max_history_messages:]
        for message in recent_history:
            messages.append(message)
        
        messages.append({"role": "user", "content": user_input})

        # Add user message to chat history immediately
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input}
        )
        
        # Display user message (will trigger auto-scroll)
        with st.container():
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Get and stream response
        response = call_deepseek_api(
            messages=messages, temperature=temperature
        )
        
        if response:
            full_response = ""
            response_placeholder = st.empty()
            
            # Stream the response without rerunning mid-stream
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    st.session_state["temp_response"] = full_response
                    with response_placeholder.container():
                        with st.chat_message("assistant"):
                            st.markdown(full_response + "▌")
            
            # Finalize the response
            del st.session_state["temp_response"]
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": full_response}
            )
            
            # Only rerun after the response is fully streamed
            st.rerun()
        else:
            st.warning("No response received from the API.")