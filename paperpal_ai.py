import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Custom CSS for PaperPal AI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0E1117, #1E1E1E);
    color: #FFFFFF;
    font-family: 'Arial', sans-serif;
}
h1 {
    color: #00FFAA;
    text-align: center;
    font-size: 3rem;
    margin-bottom: 20px;
}
h3 {
    color: #00FFAA;
    text-align: center;
    font-size: 1.5rem;
    margin-bottom: 30px;
}
.stFileUploader {
    background-color: #1E1E1E;
    border: 2px dashed #00FFAA;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.stFileUploader:hover {
    background-color: #2A2A2A;
    border-color: #00FFFF;
}
.stChatInput input {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    border: 1px solid #3A3A3A !important;
    border-radius: 5px;
    padding: 10px;
    transition: all 0.3s ease;
}
.stChatInput input:focus {
    border-color: #00FFAA !important;
    box-shadow: 0 0 5px #00FFAA;
}
.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #1E1E1E !important;
    border: 1px solid #3A3A3A !important;
    color: #E0E0E0 !important;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #2A2A2A !important;
    border: 1px solid #404040 !important;
    color: #F0F0F0 !important;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.avatar {
    background-color: #00FFAA !important;
    color: #000000 !important;
}
.stSpinner {
    color: #00FFAA !important;
}
</style>
""", unsafe_allow_html=True)

# Constants
PROMPT_TEMPLATE = """
You are an expert research assistant named PaperPal AI. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).
Query: {user_query}
Context: {document_context}
Answer:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/'
TEXT_STORAGE_PATH = 'document_store/texts/'
DOCX_STORAGE_PATH = 'document_store/docx/'
HISTORY_STORAGE_PATH = 'history/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Ensure directories exist
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
os.makedirs(TEXT_STORAGE_PATH, exist_ok=True)
os.makedirs(DOCX_STORAGE_PATH, exist_ok=True)
os.makedirs(HISTORY_STORAGE_PATH, exist_ok=True)

# Helper Functions
def save_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        storage_path = PDF_STORAGE_PATH
    elif uploaded_file.type == "text/plain":
        storage_path = TEXT_STORAGE_PATH
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        storage_path = DOCX_STORAGE_PATH
    else:
        return None
    file_path = os.path.join(storage_path, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_documents(file_path, file_type):
    if file_type == "pdf":
        loader = PDFPlumberLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    return loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    try:
        return DOCUMENT_VECTOR_DB.similarity_search(query)
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    try:
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I'm sorry, but I couldn't process your request at this time."

def save_conversation_history(user_query, ai_response, regenerate_count=0):
    history_file = os.path.join(HISTORY_STORAGE_PATH, "conversation_history.txt")
    with open(history_file, "a") as file:
        file.write(f"User: {user_query}\nAI: {ai_response}\nRegenerated: {regenerate_count} times\n{'-'*50}\n")

def download_history():
    history_file = os.path.join(HISTORY_STORAGE_PATH, "conversation_history.txt")
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            st.download_button(
                label="📄 Download History",
                data=file.read(),
                file_name="conversation_history.txt",
                mime="text/plain"
            )

# UI Configuration
st.title("📚 PaperPal AI")
st.markdown("### Your Intelligent Research Companion")
st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader(
    "Upload a Document (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    help="Select a document for analysis",
    accept_multiple_files=False
)

if uploaded_file:
    saved_path = save_uploaded_file(uploaded_file)
    if saved_path:
        file_type = uploaded_file.type.split('/')[-1]
        raw_docs = load_documents(saved_path, file_type)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        st.success("✅ Document processed successfully! Ask your questions below.")
    else:
        st.error("❌ Unsupported file type. Please upload a PDF, TXT, or DOCX file.")

# Chat Interface
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "regenerate_count" not in st.session_state:
    st.session_state.regenerate_count = {}

if "error_occurred" not in st.session_state:
    st.session_state.error_occurred = False

user_input = st.chat_input("Ask PaperPal AI about the document...")
if user_input:
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    with st.spinner("🤖 Analyzing document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)

        if "Error generating answer" in ai_response:
            st.session_state.error_occurred = True
        else:
            st.session_state.error_occurred = False

        save_conversation_history(user_input, ai_response, st.session_state.regenerate_count.get(user_input, 0))
        st.session_state.conversation_history.append((user_input, ai_response))

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)

# Show Previous Questions and Answers
for i, (question, answer) in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5 questions
    with st.chat_message("user", avatar="👤"):
        st.write(question)
    with st.chat_message("assistant", avatar="🤖"):
        st.write(answer)
    st.markdown("---")

# Reload Answer Button
if st.session_state.conversation_history:
    last_question, _ = st.session_state.conversation_history[-1]
    if st.button("🔄 Regenerate Last Answer"):
        regenerate_count = st.session_state.regenerate_count.get(last_question, 0) + 1
        st.session_state.regenerate_count[last_question] = regenerate_count

        with st.spinner(f"🔄 Regenerating answer ({regenerate_count} times)..."):
            relevant_docs = find_related_documents(last_question)
            new_ai_response = generate_answer(last_question, relevant_docs)
            st.session_state.conversation_history[-1] = (last_question, f"({regenerate_count}) {new_ai_response}")
            save_conversation_history(last_question, new_ai_response, regenerate_count)

        # Display the regenerated answer in the chat interface
        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"({regenerate_count}) {new_ai_response}")
        st.success(f"✅ Answer regenerated! Total regenerations: {regenerate_count}")

# Error Handling Retry Button
if st.session_state.error_occurred:
    if st.button("⚠️ Try Again After Error"):
        last_question, _ = st.session_state.conversation_history[-1]
        with st.spinner("🔄 Retrying last operation..."):
            relevant_docs = find_related_documents(last_question)
            new_ai_response = generate_answer(last_question, relevant_docs)
            st.session_state.conversation_history[-1] = (last_question, new_ai_response)
            st.success("✅ Operation retried successfully!")

# Download History Button
if st.session_state.conversation_history:
    download_history()