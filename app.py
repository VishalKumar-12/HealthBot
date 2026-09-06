import os
import streamlit as st

from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="HealthBot",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# API Keys
# =========================================================

PINECONE_API_KEY = st.secrets.get(
    "PINECONE_API_KEY",
    os.getenv("PINECONE_API_KEY")
)

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY is missing.")
    st.stop()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing.")
    st.stop()


# =========================================================
# Load Retriever
# =========================================================

@st.cache_resource
def get_retriever():

    embedding = download_embeddings()

    vector_store = PineconeVectorStore(
        index_name="healthbot-multilingual-v2-384",
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )


# =========================================================
# Load LLM
# =========================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY,
        temperature=0
    )


# =========================================================
# Prompt
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# =========================================================
# Load resources
# =========================================================

try:

    retriever = get_retriever()
    llm = get_llm()

except Exception as e:

    st.error("HealthBot could not start.")
    st.exception(e)
    st.stop()


# =========================================================
# UI
# =========================================================

st.title("🩺 HealthBot")

st.write(
    "AI Medical Assistant powered by your medical knowledge system."
)


# =========================================================
# Session History
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# Display previous messages
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

    # -----------------------------------------------------
    # User message
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # -----------------------------------------------------
    # Greeting
    # -----------------------------------------------------

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
        "namaste"
    ]

    if user_input.lower() in greetings:

        answer = "👋 Hello! I am HealthBot. How can I help you?"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.stop()


    # -----------------------------------------------------
    # Search Pinecone
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching medical information..."):

            try:

                docs = retriever.invoke(user_input)

            except Exception as e:

                st.error(
                    "Sorry, the medical search service "
                    "is temporarily unavailable."
                )

                st.stop()


            if not docs:

                answer = (
                    "I don't have information about this topic."
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.stop()


            # -------------------------------------------------
            # Create context
            # -------------------------------------------------

            context = "\n\n".join(
                f"[PDF Page {doc.metadata.get('page', 0) + 1}]\n"
                f"{doc.page_content}"
                for doc in docs
            )


            # -------------------------------------------------
            # Generate answer
            # -------------------------------------------------

            messages = prompt.format_messages(
                context=context,
                input=user_input
            )

            try:

                response = llm.invoke(messages)

                answer = response.content

            except Exception:

                answer = (
                    "Sorry, I am unable to generate "
                    "an answer right now."
                )


            # -------------------------------------------------
            # Display answer
            # -------------------------------------------------

            st.markdown(answer)


            # -------------------------------------------------
            # Sources
            # -------------------------------------------------

            sources = []

            for doc in docs:

                page = int(
                    doc.metadata.get("page", 0)
                ) + 1

                source = doc.metadata.get(
                    "source",
                    ""
                )

                if source:

                    filename = os.path.basename(source)

                    sources.append(
                        f"📄 **Page {page}** — `{filename}`"
                    )


            sources = list(
                dict.fromkeys(sources)
            )


            if sources:

                with st.expander("📖 Sources"):

                    for source in sources:
                        st.markdown(source)


    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })