# Document Chat Interface with DeepSeek API

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-000000?style=for-the-badge&logo=openai&logoColor=white)

A **Streamlit-based web application** that allows users to interact with documents using the **DeepSeek API**. Upload files (PDF, DOCX, TXT), ask questions, and receive answers based on the file's content. The app supports **streaming responses**, **context-aware answers**, and **customizable settings** like temperature and default prompts.

---

## Key Features

- **File Upload**: Supports PDF, DOCX, and TXT files for context-based queries.
- **Streaming Responses**: Real-time, incremental responses for a smooth user experience.
- **Context-Aware Answers**: Combines default context, user prompts, and file content for accurate responses.
- **Customizable Settings**:
  - Set **default prompts** for system behavior.
  - Adjust **temperature** for response creativity (low for accuracy, high for creativity).
- **Chat History**: Displays conversation history grouped by date (Today, Yesterday, 7 Days, Older).
- **Task-Specific Temperature**: Automatically adjusts temperature based on task type (e.g., coding, general questions, creative tasks).

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

