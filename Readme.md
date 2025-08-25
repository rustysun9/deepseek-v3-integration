# DeepSeek Chat Interface V3.1

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-000000?style=for-the-badge&logo=openai&logoColor=white)

A **Streamlit-based web application** that provides advanced chat capabilities using **DeepSeek's latest models** including the new **deepseek-reasoner** with chain-of-thought reasoning. Supports multi-file context, intelligent token management, and professional-grade document processing.

---

## 🚀 Key Features (V3.1)

### 🤖 Dual Model Support
- **deepseek-chat**: Standard conversational model (8K output tokens)
- **deepseek-reasoner**: Advanced reasoning model with **chain-of-thought** (64K output tokens)
- Real-time reasoning content display for deepseek-reasoner

### 📁 Enhanced File Support
**Supported File Types:**
- **Documents**: PDF, TXT, DOCX, MD
- **Code**: Python (.py), C/C++ (.c, .cpp, .h), Arduino (.ino), JavaScript (.js), TypeScript (.ts)
- **Web**: HTML, CSS, PHP
- **Config**: JSON, YAML, XML, INI, CFG
- **System**: Makefiles (.mk), Bash (.sh), Systemd services (.service)
- **Templates**: Jinja2 (.j2), Apache configs (.conf)
- **Networking**: NetworkManager connections (.nmconnection)

### 🧠 Intelligent Context Management
- **Smart token truncation** preserving code structure and important content
- **Context-aware prioritization** (imports > classes > functions > comments)
- **Dynamic system message** generation combining default prompt + file context
- **API-enforced limits**: 128K input tokens with intelligent enforcement

### 💾 Advanced Chat Management
- **Save/Load chats** with metadata (name, description, timestamps)
- **Chat history persistence** with JSON storage
- **One-click chat deletion** with confirmation
- **Automatic refresh** of chat lists

### ⚡ Performance Features
- **Streaming responses** with real-time typing simulation
- **Cache efficiency metrics** showing hit/miss rates
- **Token usage progress bars** with real-time updates
- **Balance tracking** with DeepSeek API integration

### 🎛️ Professional Settings
- **Task-specific temperature ranges**:
  - Coding/Math: 0.0-0.3 (precision)
  - Data Analysis: 0.3-0.7 (balanced)
  - Normal Questions: 0.5-0.9 (conversational)  
  - Creative Writing: 0.8-1.5 (creative)
- **Default prompt management** with JSON persistence
- **Model-specific constraints** (temperature disabled for reasoning model)

### 🔧 Technical Excellence
- **Robust error handling** with comprehensive try-catch blocks
- **File encoding detection** using charset_normalizer
- **Atomic file operations** with temp files for data integrity
- **Memory-efficient file processing** with chunked reading

---

## 🛠️ Technologies Used

- **Streamlit** - Web interface framework
- **DeepSeek API** - AI model integration (v3.1 models)
- **PyMuPDF (fitz)** - PDF text extraction
- **charset_normalizer** - File encoding detection
- **tiktoken** - Token counting and management
- **python-decouple** - Environment configuration

---

## How to Use

1. Clone the repository:
   ```bash
   git clone git@github.com:rustysun9/deepseek-v3-integration.git

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Set up your .env file with your DeepSeek API key::
    ```bash
    touch .env

4.  Open the `.env` file and add your DeepSeek API key:

    ```bash
    DEEPSEEK_API_KEY=your_api_key_here

5. Run the Streamlit app:
    ```bash
    streamlit run main.py

