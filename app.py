import os
import urllib.parse
import streamlit as st

from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Environment / memory optimization
# =========================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


st.set_page_config(
    page_title="HealthBot",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# API Keys
# =========================================================

try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not PINECONE_API_KEY or not GROQ_API_KEY:
    st.error("API keys are missing. Please add them in Streamlit Secrets.")
    st.stop()


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []


# =========================================================
# Cached Embeddings
# =========================================================

@st.cache_resource(show_spinner="Loading HealthBot...")
def get_embeddings():
    return download_embeddings()


# =========================================================
# Cached Retriever
# =========================================================

@st.cache_resource(show_spinner="Connecting to medical database...")
def get_retriever():

    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(
        index_name="healthbot-multilingual-v2-384",
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 2
        }
    )


# =========================================================
# Cached LLM
# =========================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY,
        temperature=0
    )


# =========================================================
# Greeting
# =========================================================

def is_greeting(text):

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "namaste",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    text = text.lower().strip()

    return text in greetings


# =========================================================
# Header
# =========================================================

st.title("🩺 HealthBot")
st.caption("AI Medical Assistant powered by RAG, Pinecone & Groq")


# =========================================================
# Chat History
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# User Input
# =========================================================

user_input = st.chat_input(
    "Ask your medical question..."
)


if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Keep only recent messages
    st.session_state.messages = st.session_state.messages[-10:]

    # =====================================================
    # Greeting
    # =====================================================

    if is_greeting(user_input):

        answer = (
            "Hello! 👋 I am HealthBot.\n\n"
            "I can answer medical questions using the "
            "medical knowledge available in my database."
        )

        st.session_state.sources = []

    else:

        # =================================================
        # Load resources only when actually needed
        # =================================================

        retriever = get_retriever()
        llm = get_llm()

        # =================================================
        # Retrieve documents
        # =================================================

        docs = retriever.invoke(user_input)

        # Save only lightweight source information
        sources = []

        for doc in docs:

            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", 0)

            pdf_name = os.path.basename(source)

            try:
                page_number = int(page) + 1
            except Exception:
                page_number = 1

            sources.append(
                {
                    "pdf": pdf_name,
                    "page": page_number
                }
            )

        st.session_state.sources = sources

        # =================================================
        # Build context
        # =================================================

        context_parts = []

        for doc in docs:

            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", 0)

            pdf_name = os.path.basename(source)

            try:
                page_number = int(page) + 1
            except Exception:
                page_number = 1

            context_parts.append(
                f"[PDF: {pdf_name} | Page {page_number}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # =================================================
        # Prompt
        # =================================================

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                (
                    "human",
                    "Medical context:\n\n{context}\n\n"
                    "User question:\n{question}"
                )
            ]
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "context": context,
                "question": user_input
            }
        )

        answer = response.content

    # =====================================================
    # Assistant Response
    # =====================================================

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Keep history small
    st.session_state.messages = st.session_state.messages[-10:]


# =========================================================
# Sources
# =========================================================

if st.session_state.sources:

    st.divider()

    st.subheader("📚 Sources")

    displayed = set()

    for source in st.session_state.sources:

        pdf_name = source["pdf"]
        page = source["page"]

        key = (pdf_name, page)

        if key in displayed:
            continue

        displayed.add(key)

        encoded_name = urllib.parse.quote(
            pdf_name,
            safe=""
        )

        pdf_url = (
            "https://github.com/"
            "VishalKumar-12/HealthBot/"
            "blob/main/data/"
            f"{encoded_name}"
            f"#page={page}"
        )

        st.link_button(
            f"📖 {pdf_name} — Page {page}",
            pdf_url,
            use_container_width=True
        )