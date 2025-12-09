"""
RAG Q&A Chatbot with Streamlit UI using Groq API
Requirements: pip install streamlit langchain langchain-groq langchain-community chromadb sentence-transformers pypdf python-dotenv
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def load_documents(uploaded_files):
    """Load documents from uploaded files"""
    documents = []

    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"Skipped {uploaded_file.name}: Unsupported format")
                continue

            docs = loader.load()
            documents.extend(docs)
            st.success(f"✅ Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)

    return documents

def create_vectorstore(documents, embeddings):
    """Create vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore, len(chunks)

def setup_qa_chain(llm, vectorstore):
    """Setup QA chain"""
    prompt_template = """Use the following context to answer the question. 
    If you don't know the answer based on the context, say "I don't have enough information to answer that question based on the provided documents."
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def main():
    # Title and description
    st.title("🤖 RAG Q&A Chatbot with Groq")
    st.markdown("Upload your documents (PDF/TXT) and ask questions!")

    # Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Display API key status
        if groq_api_key:
            st.success("✅ API Key loaded from .env")
        else:
            st.error("❌ API Key not found in .env file")
            st.info("Please add GROQ_API_KEY to your .env file")

        # Model selection
        model_name = st.selectbox(
            "Select Model",
            [
                "llama-3.3-70b-versatile",      # Most capable
                "llama-3.1-8b-instant",         # Fast and efficient
                "qwen/qwen3-32b",               # Good alternative
                "openai/gpt-oss-20b"            # OpenAI's open model
            ],
            help="Choose the Groq model for answers"
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values make output more creative"
        )

        st.divider()

        # File upload
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )

        # Process documents button
        if uploaded_files and groq_api_key:
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        # Initialize embeddings
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )

                        # Load documents
                        documents = load_documents(uploaded_files)

                        if documents:
                            # Create vector store
                            vectorstore, num_chunks = create_vectorstore(documents, embeddings)
                            st.session_state.vectorstore = vectorstore

                            # Initialize Groq LLM
                            llm = ChatGroq(
                                groq_api_key=groq_api_key,
                                model_name=model_name,
                                temperature=temperature
                            )

                            # Setup QA chain
                            qa_chain = setup_qa_chain(llm, vectorstore)
                            st.session_state.qa_chain = qa_chain
                            st.session_state.documents_processed = True

                            st.success(f"✅ Processed {len(documents)} document(s) into {num_chunks} chunks!")
                        else:
                            st.error("No documents were successfully loaded")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        elif uploaded_files and not groq_api_key:
            st.warning("⚠️ Please add GROQ_API_KEY to your .env file")

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        # Reset all button
        if st.button("🔄 Reset All"):
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = False
            st.rerun()

    # Main chat interface
    if not groq_api_key:
        st.info("⚠️ Please configure your .env file with GROQ_API_KEY")
        st.markdown("""
        ### How to get started:
        1. Create a `.env` file in the project root
        2. Add your Groq API key: `GROQ_API_KEY=your_api_key_here`
        3. Get your API key from [Groq Console](https://console.groq.com/)
        4. Restart the application
        5. Upload your PDF or TXT documents
        6. Start asking questions!
        """)
        return

    if not st.session_state.documents_processed:
        st.info("👈 Please upload and process documents in the sidebar")
        return

    # Display chat history
    st.subheader("💬 Chat")

    # Chat container
    chat_container = st.container()

    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

                # Show sources in expander
                if chat.get("sources"):
                    with st.expander("📚 View Sources"):
                        for j, source in enumerate(chat["sources"], 1):
                            st.markdown(f"**Source {j}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
                            st.divider()

    # Question input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": question})
                    answer = result["result"]
                    source_docs = result["source_documents"]

                    st.write(answer)

                    # Show sources
                    if source_docs:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.divider()

                    # Save to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": [doc.page_content for doc in source_docs]
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()