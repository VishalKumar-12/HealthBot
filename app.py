
import os
import urllib.parse

import streamlit as st

from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HealthBot",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# API KEYS
# =========================================================

try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

except Exception:

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not PINECONE_API_KEY or not GROQ_API_KEY:

    st.error("API key is missing.")

    st.stop()


# =========================================================
# RETRIEVER
# =========================================================

@st.cache_resource
def get_retriever():

    embedding = download_embeddings()

    vector_store = PineconeVectorStore(
        index_name="healthbot-multilingual-v2-384",
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return retriever


# =========================================================
# LLM
# =========================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY,
        temperature=0
    )


# =========================================================
# INITIALIZE
# =========================================================

retriever = get_retriever()
llm = get_llm()


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" not in st.session_state:
    st.session_state.docs = []


# =========================================================
# UI
# =========================================================

st.title("🩺 HealthBot")

st.write(
    "AI Medical Assistant — Ask questions about your medical documents."
)


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"],
            unsafe_allow_html=True
        )


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask your medical question..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if user_input:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):

        st.markdown(user_input)


    # -----------------------------------------------------
    # GREETINGS
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


    if user_input.lower().strip() in greetings:

        answer = (
            "👋 Hello! I am HealthBot. "
            "How can I help you with your medical questions?"
        )

        st.session_state.docs = []

        with st.chat_message("assistant"):

            st.markdown(answer)


    # -----------------------------------------------------
    # MEDICAL QUESTION
    # -----------------------------------------------------

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching medical documents..."
            ):

                try:

                    # Retrieve documents
                    docs = retriever.invoke(user_input)


                    # Save retrieved documents
                    st.session_state.docs = docs


                    # -------------------------------------------------
                    # BUILD CONTEXT
                    # -------------------------------------------------

                    context_parts = []

                    for doc in docs:

                        page = int(
                            doc.metadata.get("page", 0)
                        ) + 1

                        context_parts.append(
                            f"[PDF Page {page}]\n"
                            f"{doc.page_content}"
                        )


                    context = "\n\n".join(
                        context_parts
                    )


                    # -------------------------------------------------
                    # CREATE PROMPT
                    # -------------------------------------------------

                    messages = prompt.format_messages(
                        context=context,
                        input=user_input
                    )


                    # -------------------------------------------------
                    # GENERATE ANSWER
                    # -------------------------------------------------

                    response = llm.invoke(messages)

                    answer = response.content


                    # -------------------------------------------------
                    # DISPLAY ANSWER
                    # -------------------------------------------------

                    st.markdown(
                        answer,
                        unsafe_allow_html=True
                    )


                except Exception as e:

                    st.error(
                        "❌ Unable to generate answer."
                    )

                    st.exception(e)

                    st.stop()


    # =====================================================
    # SOURCES
    # =====================================================

    if st.session_state.docs:

        shown = set()


        with st.expander("📖 Sources"):

            for doc in st.session_state.docs:

                # -------------------------------------------------
                # GET PAGE
                # -------------------------------------------------

                page = doc.metadata.get("page")

                if page is None:
                    continue


                try:

                    display_page = int(page) + 1

                except:

                    continue


                # -------------------------------------------------
                # PDF NAME
                # -------------------------------------------------

                pdf_path = doc.metadata.get(
                    "source",
                    "Medical_book.pdf"
                )


                pdf_path = str(pdf_path).replace(
                    "\\",
                    "/"
                )


                filename = os.path.basename(
                    pdf_path
                )


                # If metadata doesn't contain filename
                if not filename:

                    filename = "Medical_book.pdf"


                # -------------------------------------------------
                # REMOVE DUPLICATE PAGES
                # -------------------------------------------------

                key = (
                    filename,
                    display_page
                )


                if key in shown:
                    continue


                shown.add(key)


                # -------------------------------------------------
                # GITHUB RAW PDF
                # -------------------------------------------------

                pdf_url = (
                    "https://raw.githubusercontent.com/"
                    "VishalKumar-12/HealthBot/main/"
                    "data/Medical_book.pdf"
                )


                # Encode complete PDF URL
                encoded_pdf_url = urllib.parse.quote(
                    pdf_url,
                    safe=""
                )


                # -------------------------------------------------
                # PDF.JS VIEWER
                # -------------------------------------------------

                viewer_url = (
                    "https://mozilla.github.io/pdf.js/web/viewer.html"
                    f"?file={encoded_pdf_url}"
                    f"#page={display_page}"
                )


                # -------------------------------------------------
                # SOURCE CARD
                # -------------------------------------------------

                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 12px;
                        margin-bottom: 10px;
                    ">

                        <div style="
                            font-size: 16px;
                            font-weight: 600;
                            margin-bottom: 8px;
                        ">
                            📄 {filename}
                        </div>

                        <div style="
                            margin-bottom: 10px;
                            color: #666;
                        ">
                            Page {display_page}
                        </div>

                        <a
                            href="{viewer_url}"
                            target="_blank"
                            style="
                                text-decoration: none;
                            "
                        >

                            <button style="
                                width: 100%;
                                padding: 10px;
                                border-radius: 8px;
                                border: 1px solid #ccc;
                                background: transparent;
                                cursor: pointer;
                                font-size: 15px;
                            ">

                                📄 Open PDF — Page {display_page}

                            </button>

                        </a>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

