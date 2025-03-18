import streamlit as st
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

# Initialize session state for chat history and chain
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Streamlit app title
st.title("PDF Chatbot")

# File uploader for PDF files
uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Process each uploaded file
    texts = []
    metadatas = []
    for file in uploaded_files:
        # Read the PDF file
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
            
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Create metadata for each chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama3"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    # Store the chain in session state
    st.session_state.chain = chain

    # Inform the user that processing has ended
    st.success(f"Processing {len(uploaded_files)} files done. You can now ask questions!")

# Chat interface
user_input = st.text_input("Ask a question:")

if user_input and st.session_state.chain:
    # Retrieve the chain from session state
    chain = st.session_state.chain
    
    # Call the chain with user's message content
    res = chain.invoke({"question": user_input})
    answer = res["answer"]
    source_documents = res["source_documents"]

    # Display the answer
    st.write("Answer:", answer)

    # Process source documents if available
    if source_documents:
        st.write("Sources:")
        for source_idx, source_doc in enumerate(source_documents):
            st.write(f"Source {source_idx + 1}: {source_doc.page_content}")

# Display chat history
if st.session_state.chat_history:
    st.write("Chat History:")
    for msg in st.session_state.chat_history:
        st.write(f"{msg['role']}: {msg['content']}")
