import streamlit as st
from pdf_utils import extract_text_from_pdf, clean_text, chunk_text
from db_utils import setup_collection, insert_documents, search
from query_utils import ask_llama

st.set_page_config(page_title="Knowledge Chat", page_icon="📚")
st.title("📚 Knowledge Chat System")

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
        clean = clean_text(raw_text)
        chunks = chunk_text(clean)

        with st.spinner("Setting up database..."):
            setup_collection()
            insert_documents(chunks)
        
        st.success("✅ PDF processed and stored!")

# Chat interface
st.header("Chat with your Knowledge Base 🧠")

query = st.chat_input("Ask something about the uploaded PDF...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Thinking..."):
        relevant_chunks = search(query)
        context = "\n\n".join(relevant_chunks)
        answer = ask_llama(context, query)
    
    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display the chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
