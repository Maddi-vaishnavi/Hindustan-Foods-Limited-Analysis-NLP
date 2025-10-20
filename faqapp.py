import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# --- Configuration ---
PDF_FILE_PATH = "Hindustan Foods Limited faqs.pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-3.5-turbo"
CACHE_DIR = "./chroma_db_cache"

# --- Quick FAQ Questions with Icons ---
QUICK_FAQS = [
    {
        "icon": "🏢",
        "title": "About HFL",
        "question": "What is Hindustan Foods Limited (HFL)?"
    },
    {
        "icon": "💼",
        "title": "Business",
        "question": "What is the main business of HFL?"
    },
    {
        "icon": "📅",
        "title": "Founded",
        "question": "What year was HFL incorporated?"
    },
    {
        "icon": "🎯",
        "title": "Vision",
        "question": "What is HFL's vision?"
    },
    {
        "icon": "🚀",
        "title": "Mission",
        "question": "What is HFL's mission?"
    },
    {
        "icon": "🤝",
        "title": "Partners",
        "question": "Who are the key clients and partners of HFL?"
    }
]

# --- Helper Functions ---

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_file_path, embedding_model_name):
    """Loads PDF, splits text, creates embeddings, stores in Chroma, and returns retriever."""
    try:
        status_placeholder = st.empty()
        
        with status_placeholder.container():
            with st.spinner("Processing document... Please wait."):
                loader = PyPDFLoader(uploaded_file_path)
                docs = loader.load()
                if not docs:
                    st.error("Failed to load document. No pages found.")
                    return None

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                if not splits:
                    st.error("Failed to split document into chunks.")
                    return None

                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': False}
                )

                if not os.path.exists(CACHE_DIR):
                    os.makedirs(CACHE_DIR)

                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=CACHE_DIR
                )

                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        status_placeholder.empty()
        return retriever

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# --- Streamlit App UI ---

st.set_page_config(page_title="Hindustan Foods Limited FAQ Chatbot", layout="wide", page_icon="🍽️")
st.title("🍽️ Hindustan Foods Limited FAQ Chatbot")
st.caption("Ask questions about Hindustan Foods Limited or use the quick FAQs below!")

# Custom CSS for FAQ cards (light theme)
st.markdown("""
    <style>
    .faq-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
        color: white;
    }
    .faq-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .faq-icon {
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .faq-title {
        font-size: 1.1em;
        font-weight: 600;
        margin: 0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    openrouter_api_key = st.text_input("Enter your OpenRouter API Key:", type="password")
    st.markdown("[Get an OpenRouter API key](https://openrouter.ai/keys)")

    if os.path.exists(PDF_FILE_PATH):
        st.success(f"Using PDF: {os.path.basename(PDF_FILE_PATH)}")
        file_size = os.path.getsize(PDF_FILE_PATH) / (1024 * 1024)
        st.info(f"File Size: {file_size:.2f} MB")
    else:
        st.error(f"Error: PDF file not found at '{PDF_FILE_PATH}'. Please place the PDF in the same folder as app.py.")
        st.stop()

# --- Main Chat Interface ---

if not openrouter_api_key:
    st.info("Please enter your OpenRouter API Key in the sidebar to start.")
    st.stop()

try:
    retriever = configure_retriever(PDF_FILE_PATH, EMBEDDING_MODEL)

    if retriever:
        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        memory = ConversationBufferMemory(
            chat_memory=msgs,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )
        if len(msgs.messages) == 0:
            msgs.add_ai_message("Hello! How can I help you with Hindustan Foods Limited?")

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            max_tokens=800,
            streaming=True
        )

        # Enhanced system prompt for better responses
        system_message = """You are a helpful assistant answering questions about Hindustan Foods Limited. 
        When answering:
        - Provide concise, well-structured summaries with key points
        - Use bullet points for clarity when listing multiple items
        - Keep responses focused and clear
        - If the answer is not in the context, clearly state that"""
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )

        # --- Quick FAQs Section with Clickable Cards ---
        st.subheader("💡 Quick FAQs")
        st.caption("Click on any card to get an instant answer:")
        
        if 'faq_clicked' not in st.session_state:
            st.session_state.faq_clicked = None
        
        # Create horizontal cards (2 rows of 3)
        row1_cols = st.columns(3)
        row2_cols = st.columns(3)
        
        for idx, faq in enumerate(QUICK_FAQS[:3]):
            with row1_cols[idx]:
                if st.button(
                    f"{faq['icon']}\n\n{faq['title']}", 
                    key=f"faq_{idx}", 
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.faq_clicked = faq['question']
        
        for idx, faq in enumerate(QUICK_FAQS[3:], start=3):
            with row2_cols[idx-3]:
                if st.button(
                    f"{faq['icon']}\n\n{faq['title']}", 
                    key=f"faq_{idx}", 
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.faq_clicked = faq['question']
        
        st.divider()

        # Render current messages
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        # Handle FAQ click
        if st.session_state.faq_clicked:
            prompt = st.session_state.faq_clicked
            st.session_state.faq_clicked = None
            
            st.chat_message("human").write(prompt)
            
            with st.chat_message("ai"):
                try:
                    config = {"configurable": {"session_id": "any"}}
                    full_response = qa_chain.invoke({"question": prompt}, config=config)
                    st.write(full_response["answer"])

                    with st.expander("📄 View Sources"):
                        if "source_documents" in full_response and full_response["source_documents"]:
                            for i, doc in enumerate(full_response["source_documents"]):
                                page_num = doc.metadata.get('page', 'N/A')
                                st.markdown(f"**Source {i+1} (Page: {page_num})**")
                                st.caption(doc.page_content[:400] + "...")
                        else:
                            st.write("No source documents found for this response.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Accept new user input
        if prompt := st.chat_input("Ask a question about Hindustan Foods Limited..."):
            st.chat_message("human").write(prompt)

            config = {"configurable": {"session_id": "any"}}
            with st.chat_message("ai"):
                try:
                    full_response = qa_chain.invoke({"question": prompt}, config=config)
                    st.write(full_response["answer"])

                    with st.expander("📄 View Sources"):
                        if "source_documents" in full_response and full_response["source_documents"]:
                            for i, doc in enumerate(full_response["source_documents"]):
                                page_num = doc.metadata.get('page', 'N/A')
                                st.markdown(f"**Source {i+1} (Page: {page_num})**")
                                st.caption(doc.page_content[:400] + "...")
                        else:
                            st.write("No source documents found for this response.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

except Exception as e:
    st.error(f"An unexpected error occurred during setup: {e}")
    st.info("Please ensure your OpenRouter API key is correct and the PDF file is accessible.")
