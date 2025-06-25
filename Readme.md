# Document Chat Interface with DeepSeek API

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-000000?style=for-the-badge&logo=openai&logoColor=white)

A **Streamlit-based web application** that allows users to interact with documents using the **DeepSeek API**. Upload files (PDF, DOCX, TXT), ask questions, and receive answers based on the file's content. The app supports **streaming responses**, **context-aware answers**, and **customizable settings** like temperature and default prompts.

---

# Key Features

# File Upload:
- Supports a wide range of file types, including PDF, DOCX, TXT, Python (.py), C/C++ (.c, .cpp, .h), Arduino (.ino), JavaScript (.js), HTML, CSS, JSON, YAML, and more.
- Uploaded files are processed for context-based queries.

# Dynamic Default Prompt Generation:
- Automatically generates a default prompt based on the uploaded files' types and content.
- Ensures accurate and relevant responses.

# Streaming Responses:
- Provides real-time, incremental responses for a smooth and interactive user experience.

# Context-Aware Answers:
- Combines default context, user prompts, and file content to deliver precise and contextually relevant answers.

# Customizable Settings:
- Set default prompts to define system behavior.
- Adjust temperature for response creativity (low for accuracy, high for creativity).
- Select task-specific models (e.g., coding, reasoning, or general chat).

# Chat History Management:
- Save and load chat sessions with metadata (name, description, and timestamps).
- View saved chats grouped by date (Today, Yesterday, Last 7 Days, Older).
- Delete saved chats directly from the interface.

# Task-Specific Temperature Control:
- Automatically adjusts temperature ranges based on the selected task type (e.g., coding/math, general questions, data analysis, or creative writing).

# Token Usage Management:
- Displays real-time token usage progress to ensure compliance with API limits.
- Implements token truncation strategies to prioritize important context and maintain efficiency.

# Advanced File Context Handling:
- Groups related files (e.g., .ino with corresponding .h files) for better project context.
- Truncates large files intelligently while preserving structure and relevance.

# Cache Efficiency Display:
- Shows cache hit/miss statistics to optimize API usage and improve performance.

# User Balance Information:
- Displays account balance details, including total balance, granted balance, and topped-up balance.

---

## Technologies Used

- **Streamlit**: For building the web interface.
- **DeepSeek API**: For generating responses using the `deepseek-chat` model.
- **PyMuPDF (fitz)**: For extracting text from PDF files.
- **chardet**: For detecting file encoding.
- **python-decouple**: For managing environment variables.

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

