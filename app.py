import os
import urllib.parse

import streamlit as st
from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIG
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
    st.error("❌ API key is missing.")
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
# LOAD MODELS
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
st.write("AI Medical Assistant")


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask your medical question..."
)


# =========================================================
# PROCESS USER QUESTION
# =========================================================

if user_input:

    # Default answer.
    # This prevents NameError.
    answer = ""

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

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


    # =====================================================
    # GREETING
    # =====================================================

    if user_input.lower().strip() in greetings:

        answer = (
            "👋 Hello! I am HealthBot. "
            "How can I help you?"
        )

        st.session_state.docs = []

        with st.chat_message("assistant"):
            st.markdown(answer)


    # =====================================================
    # MEDICAL QUESTION
    # =====================================================

    else:

        with st.chat_message("assistant"):

            try:

                # -----------------------------------------
                # RETRIEVE DOCUMENTS
                # -----------------------------------------

                with st.spinner(
                    "🔎 Searching medical documents..."
                ):

                    docs = retriever.invoke(user_input)


                # Save retrieved documents
                st.session_state.docs = docs


                # -----------------------------------------
                # CHECK RETRIEVED DOCUMENTS
                # -----------------------------------------

                if not docs:

                    answer = (
                        "Sorry, I could not find relevant "
                        "information in the medical documents."
                    )

                    st.warning(answer)


                else:

                    # -------------------------------------
                    # CREATE CONTEXT
                    # -------------------------------------

                    context_parts = []

                    for doc in docs:

                        # Page
                        page = doc.metadata.get(
                            "page",
                            0
                        )

                        try:
                            page_number = int(page) + 1

                        except Exception:
                            page_number = 1


                        # PDF filename
                        pdf_name = (
                            doc.metadata.get("source")
                            or doc.metadata.get("file_name")
                            or doc.metadata.get("filename")
                        )


                        # If metadata doesn't contain filename
                        if not pdf_name:
                            pdf_name = "Medical_book.pdf"


                        # Extract filename from full path
                        pdf_name = os.path.basename(
                            str(pdf_name)
                        )


                        # Context
                        context_parts.append(
                            f"[PDF: {pdf_name} | "
                            f"Page {page_number}]\n"
                            f"{doc.page_content}"
                        )


                    context = "\n\n".join(
                        context_parts
                    )


                    # -------------------------------------
                    # CREATE PROMPT
                    # -------------------------------------

                    messages = prompt.format_messages(
                        context=context,
                        input=user_input
                    )


                    # -------------------------------------
                    # GENERATE ANSWER
                    # -------------------------------------

                    with st.spinner(
                        "🤖 Generating answer..."
                    ):

                        response = llm.invoke(
                            messages
                        )

                        answer = response.content


                    # -------------------------------------
                    # DISPLAY ANSWER
                    # -------------------------------------

                    st.markdown(answer)


            # ---------------------------------------------
            # ERROR HANDLING
            # ---------------------------------------------

            except Exception as e:

                answer = (
                    "❌ Sorry, I was unable to "
                    "generate an answer."
                )

                st.error(answer)

                # Development ke time useful
                st.exception(e)

                st.session_state.docs = []


    # =====================================================
    # SOURCES
    # =====================================================

    if st.session_state.docs:

        # Unique combination:
        # PDF filename + page number
        shown_sources = set()


        with st.expander("📖 Sources"):

            for doc in st.session_state.docs:

                # -----------------------------------------
                # GET PAGE
                # -----------------------------------------

                page = doc.metadata.get("page")


                if page is None:
                    continue


                try:

                    display_page = int(page) + 1

                except Exception:

                    continue


                # -----------------------------------------
                # GET PDF NAME
                # -----------------------------------------

                pdf_name = (
                    doc.metadata.get("source")
                    or doc.metadata.get("file_name")
                    or doc.metadata.get("filename")
                )


                # Fallback
                if not pdf_name:

                    pdf_name = "Medical_book.pdf"


                # Remove folders/path
                pdf_name = os.path.basename(
                    str(pdf_name)
                )


                # -----------------------------------------
                # UNIQUE SOURCE
                # -----------------------------------------

                source_key = (
                    pdf_name,
                    display_page
                )


                if source_key in shown_sources:
                    continue


                shown_sources.add(
                    source_key
                )


                # -----------------------------------------
                # GITHUB PDF URL
                # -----------------------------------------

                encoded_pdf_name = (
                    urllib.parse.quote(
                        pdf_name,
                        safe=""
                    )
                )


                pdf_url = (
                    "https://raw.githubusercontent.com/"
                    "VishalKumar-12/HealthBot/main/"
                    f"data/{encoded_pdf_name}"
                )


                # -----------------------------------------
                # PDF.JS VIEWER
                # -----------------------------------------

                viewer_url = (
                    "https://mozilla.github.io/pdf.js/web/viewer.html"
                    f"?file={urllib.parse.quote(pdf_url, safe='')}"
                    f"#page={display_page}"
                )


                # -----------------------------------------
                # SOURCE INFORMATION
                # -----------------------------------------

                st.markdown(
                    f"📄 **{pdf_name}**"
                )

                st.caption(
                    f"Page {display_page}"
                )


                # -----------------------------------------
                # OPEN PDF BUTTON
                # -----------------------------------------

                st.link_button(
                    f"📖 Open {pdf_name} — "
                    f"Page {display_page}",
                    viewer_url,
                    use_container_width=True
                )


    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    if answer:

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })