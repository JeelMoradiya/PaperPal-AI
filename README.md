# PaperPal AI

## Overview
PaperPal AI is an intelligent research assistant built using Streamlit, LangChain, and Ollama. It allows users to upload documents (PDF, TXT, DOCX), processes them into chunks, indexes them in a vector database, and answers user queries based on the document's content. The application also includes features like conversation history tracking, answer regeneration, and error handling

## Code Breakdown
1. Imports and Setup
    Streamlit : Used for building the web interface.
    LangChain : Provides tools for document loading, splitting, embedding, and querying.
    Ollama : Embedding and language models for document processing and query answering.
2. Custom CSS
    The app uses custom CSS to style the UI, including background gradients, hover effects, and chat message styling.
3. Constants
    Paths for storing uploaded files and conversation history.
    Prompt template for generating answers.
    Embedding and language models (deepseek-r1:1.5b).

## Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements
   ```
3. Requirements dependencies:
   ```bash
   streamlit 
   langchain 
   langchain_community
   langchain_ollama
   pdfplumber
   ```
4. Run the app:
  ## Virtual Environments :
    ```bash
    streamlit run paperpal_ai.py
    ```
    or

    ```bash
    python -m streamlit run paperpal_ai.py
    ```


5. Usage
   ## Upload a Document :
     Use the file uploader to upload a PDF, TXT, or DOCX file.
     Wait for the document to be processed.
   ## Ask Questions :
     Type your question in the chat input box.
     The AI will analyze the document and provide an answer.
   ## Regenerate Answers :
     If you're unsatisfied with the AI's response, click the "Regenerate Last Answer" button.
   ## Download History :
     Click the "Download History" button to save your chat history.
