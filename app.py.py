import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖", layout="centered")

# --- Header ---
st.title("🤖 Zyro Dynamics HR Help Desk")
st.markdown("Ask any HR-related queries about leave policies, payroll, benefits, and more!")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    
    st.markdown("""
    ### About
    This chatbot answers employee questions using Zyro Dynamics internal HR policy documents.
    """)

# Ensure API keys are provided
if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# --- Initialize RAG Pipeline (Cached) ---
@st.cache_resource(show_spinner="Loading HR Documents & Building Knowledge Base...")
def init_rag_pipeline():
    # 1. Load Documents
    # Since you uploaded the PDFs to the main folder, we tell it to look in "." (the current folder)
    pdf_folder_path = "." 
    
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()

    # 2. Chunk Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Build Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
    
    # 4. RAG Chain setup
    template = """You are an expert HR Help Desk Assistant for Zyro Dynamics.
Answer the employee's question strictly using ONLY the provided context.
Keep your answer direct, accurate, and concise. Do not add any extra fluff, external knowledge, or conversational filler.
If the question asks about a policy or topic not explicitly covered in the context, you MUST politely refuse by stating EXACTLY:
"I can only answer HR-related questions from Zyro Dynamics policy documents."

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Using the new updated Groq model!
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Note: We return both chain and retriever so we can fetch sources for UI
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

try:
    rag_chain, retriever = init_rag_pipeline()
except Exception as e:
    st.error(f"Failed to initialize pipeline: {e}")
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am the Zyro Dynamics HR bot. How can I help you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What is the paternity leave policy?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            # Get response from RAG chain
            response = rag_chain.invoke(prompt)
            
            # Retrieve source documents for citations
            source_docs = retriever.invoke(prompt)
            
            # Display response
            st.markdown(response)
            
            # Display citations if not refused
            if source_docs and "I can only answer HR-related questions" not in response:
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(source_docs):
                        # Extract filename from metadata
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                        st.markdown(f"**Source {i+1}: {source_file}**")
                        st.caption(f"{doc.page_content[:300]}...")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
