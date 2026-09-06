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


# Check API keys
if not PINECONE_API_KEY or not GROQ_API_KEY:

    st.error("❌ API key is missing.")

    st.info(
        "Please add PINECONE_API_KEY and GROQ_API_KEY "
        "to Streamlit Secrets or environment variables."
    )

    st.stop()


# =========================================================
# RETRIEVER
# =========================================================

@st.cache_resource
def get_retriever():

    # Load embedding model
    embedding = download_embeddings()

    # Connect Pinecone
    vector_store = PineconeVectorStore(
        index_name="healthbot-multilingual-v2-384",
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY
    )

    # Create retriever
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
# INITIALIZE MODELS
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
# HEADER
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
# QUESTION PROCESSING
# =========================================================

if user_input:

    # -----------------------------------------------------
    # ADD USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
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


    # -----------------------------------------------------
    # GREETING RESPONSE
    # -----------------------------------------------------

    if user_input.lower().strip() in greetings:

        answer = (
            "👋 Hello! I am HealthBot. "
            "How can I help you with your medical questions?"
        )

        # Clear sources for greeting
        st.session_state.docs = []

        with st.chat_message("assistant"):

            st.markdown(answer)


    # -----------------------------------------------------
    # MEDICAL QUESTION
    # -----------------------------------------------------

    else:

        with st.chat_message("assistant"):

            try:

                # -----------------------------------------
                # SEARCH DOCUMENTS
                # -----------------------------------------

                with st.spinner(
                    "🔎 Searching medical documents..."
                ):

                    docs = retriever.invoke(user_input)


                # Save documents
                st.session_state.docs = docs


                # -----------------------------------------
                # BUILD CONTEXT
                # -----------------------------------------

                context_parts = []


                for doc in docs:

                    page = doc.metadata.get(
                        "page",
                        0
                    )

                    try:
                        display_page = int(page) + 1
                    except:
                        display_page = 1


                    context_parts.append(
                        f"[PDF Page {display_page}]\n"
                        f"{doc.page_content}"
                    )


                context = "\n\n".join(
                    context_parts
                )


                # -----------------------------------------
                # CREATE PROMPT
                # -----------------------------------------

                messages = prompt.format_messages(
                    context=context,
                    input=user_input
                )


                # -----------------------------------------
                # GENERATE ANSWER
                # -----------------------------------------

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    response = llm.invoke(messages)

                    answer = response.content


                # -----------------------------------------
                # DISPLAY ANSWER
                # -----------------------------------------

                st.markdown(
                    answer,
                    unsafe_allow_html=True
                )


            except Exception as e:

                answer = (
                    "❌ Sorry, I was unable to generate "
                    "an answer."
                )

                st.error(answer)

                st.exception(e)

                # Don't stop the whole app
                st.session_state.docs = []


    # =====================================================
    # SOURCES
    # =====================================================

    if st.session_state.docs:

        # Track already displayed pages
        shown_pages = set()


        with st.expander("📖 Sources"):

            for doc in st.session_state.docs:

                # -----------------------------------------
                # GET PAGE
                # -----------------------------------------

                page = doc.metadata.get("page")


                if page is None:
                    continue


                try:

                    # Pinecone/PDF pages are zero-based
                    display_page = int(page) + 1

                except:

                    continue


                # -----------------------------------------
                # REMOVE DUPLICATE PAGES
                # -----------------------------------------

                if display_page in shown_pages:
                    continue


                shown_pages.add(display_page)


                # -----------------------------------------
                # PDF FILE
                # -----------------------------------------

                filename = "Medical_book.pdf"


                # -----------------------------------------
                # GITHUB RAW PDF URL
                # -----------------------------------------

                pdf_url = (
                    "https://raw.githubusercontent.com/"
                    "VishalKumar-12/HealthBot/main/"
                    "data/Medical_book.pdf"
                )


                # -----------------------------------------
                # ENCODE PDF URL
                # -----------------------------------------

                encoded_pdf_url = urllib.parse.quote(
                    pdf_url,
                    safe=""
                )


                # -----------------------------------------
                # PDF.JS VIEWER URL
                # -----------------------------------------

                viewer_url = (
                    "https://mozilla.github.io/pdf.js/web/viewer.html"
                    f"?file={encoded_pdf_url}"
                    f"#page={display_page}"
                )


                # -----------------------------------------
                # SOURCE INFORMATION
                # -----------------------------------------

                st.markdown(
                    f"📄 **{filename}**  \n"
                    f"Page **{display_page}**"
                )


                # -----------------------------------------
                # OPEN PDF BUTTON
                # -----------------------------------------

                st.link_button(
                    f"📄 Open PDF — Page {display_page}",
                    viewer_url,
                    use_container_width=True
                )


                # Separator
                st.divider()


    # =====================================================
    # SAVE ASSISTANT RESPONSE
    # =====================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
