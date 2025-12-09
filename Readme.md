# 🤖 RAG Q&A Chatbot with Groq API

A powerful Question-Answering chatbot built with Streamlit, LangChain, and Groq's lightning-fast LLM API. Upload your documents (PDF/TXT) and get instant, accurate answers using Retrieval Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 📄 **Multi-Format Support**: Upload PDF and TXT files
- 🚀 **Lightning Fast**: Powered by Groq's ultra-fast inference
- 🔍 **RAG Technology**: Retrieval Augmented Generation for accurate answers
- 💬 **Interactive Chat UI**: Clean, intuitive Streamlit interface
- 📚 **Source Citations**: View source chunks for each answer
- 🎨 **Model Selection**: Choose from multiple Groq models
- 🔒 **Secure**: API keys stored in .env file
- 💾 **Persistent Chat**: Maintains conversation history

## 🎯 Demo

```
User: What are the working hours mentioned in the document?
Bot: The standard working hours are 9 AM to 6 PM, Monday through Friday. 
     Employees can work remotely up to 3 days per week.
```

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API Key (Get it from [Groq Console](https://console.groq.com/))
- Git (for version control)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-chatbot-groq.git
cd rag-chatbot-groq
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from [Groq Console](https://console.groq.com/)

## 🎮 Usage

### Run the Application

```bash
streamlit run rag_chatbot.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Chatbot

1. **Configure Settings** (Sidebar)
   - API key is automatically loaded from `.env`
   - Select your preferred Groq model
   - Adjust temperature (0.0 - 1.0)

2. **Upload Documents**
   - Click "Browse files" in sidebar
   - Select PDF or TXT files
   - Upload multiple files at once

3. **Process Documents**
   - Click "🔄 Process Documents" button
   - Wait for processing to complete

4. **Ask Questions**
   - Type your question in the chat input
   - Get instant answers with source citations
   - View source documents in expandable sections

## 🤖 Available Models

| Model | Speed | Best For |
|-------|-------|----------|
| `llama-3.3-70b-versatile` | ~280 t/s | Complex queries, detailed answers |
| `llama-3.1-8b-instant` | ~560 t/s | Fast responses, simple queries |
| `qwen/qwen3-32b` | ~400 t/s | General purpose |
| `openai/gpt-oss-20b` | ~1000 t/s | Very fast responses |

## 📁 Project Structure

```
Chatbot_Project/
│
├── main.py          # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in git)
├── .env.example           # Example environment file
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── sample_data.txt        # Sample data for testing
│
├── venv/                  # Virtual environment (not in git)
└── chroma_db/            # Vector database (not in git)
```

## ⚙️ Configuration

### Model Parameters

Adjust these in the sidebar:

- **Temperature** (0.0 - 1.0): Controls response creativity
  - 0.0: More focused and deterministic
  - 1.0: More creative and random

### RAG Parameters

Edit in `rag_chatbot.py`:

```python
# Chunk size for document splitting
chunk_size=1000
chunk_overlap=200

# Number of relevant chunks to retrieve
search_kwargs={"k": 3}
```

## 🔧 Troubleshooting

### API Key Not Found

```
❌ API Key not found in .env file
```

**Solution**: Ensure `.env` file exists with `GROQ_API_KEY=your_key`

### Model Decommissioned Error

```
Error: The model has been decommissioned
```

**Solution**: Update to latest models in the dropdown (already fixed in v1.1)

### Import Errors

```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**: Activate virtual environment and run `pip install -r requirements.txt`

### ChromaDB Issues

```
Error: Could not create vector store
```

**Solution**: Delete `chroma_db/` folder and reprocess documents

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for ultra-fast LLM inference
- [LangChain](https://www.langchain.com/) for RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [HuggingFace](https://huggingface.co/) for embeddings model

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/rag-chatbot-groq](https://github.com/yourusername/rag-chatbot-groq)

## 🔮 Future Enhancements

- [ ] Support for more file formats (DOCX, CSV)
- [ ] Conversation memory across sessions
- [ ] Export chat history
- [ ] Multi-language support
- [ ] Advanced filtering options
- [ ] Batch document processing

---

⭐ If you found this project helpful, please give it a star!
