import os
import urllib.parse

import streamlit as st
from src.Ingestion import download_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="HealthBot",
    page_icon="🩺",
    layout="wide"
)


# API Keys
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    st.error("API key is missing.")
    st.stop()


# Retriever
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


# LLM
@st.cache_resource
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY,
        temperature=0
    )


retriever = get_retriever()
llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# Session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" not in st.session_state:
    st.session_state.docs = []


# UI
st.title("🩺 HealthBot")
st.write("AI Medical Assistant")


# Chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_input = st.chat_input("Ask your medical question...")


if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    greetings = [
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon",
        "good evening", "namaste"
    ]


    if user_input.lower().strip() in greetings:

        answer = (
            "👋 Hello! I am HealthBot. "
            "How can I help you?"
        )

        st.session_state.docs = []

        with st.chat_message("assistant"):
            st.markdown(answer)


    else:

        with st.chat_message("assistant"):

            try:

                with st.spinner("🔎 Searching medical documents..."):

                    docs = retriever.invoke(user_input)

                st.session_state.docs = docs

                context = "\n\n".join(
                    f"[PDF Page {int(doc.metadata.get('page', 0)) + 1}]\n"
                    f"{doc.page_content}"
                    for doc in docs
                )

                messages = prompt.format_messages(
                    context=context,
                    input=user_input
                )

                with st.spinner("🤖 Generating answer..."):

                    response = llm.invoke(messages)
                    answer = response.content

                st.markdown(answer)

            except Exception as e:

                answer = "❌ Sorry, I was unable to generate an answer."

                st.error(answer)
                st.exception(e)

                st.session_state.docs = []


    # Sources
    if st.session_state.docs:

        shown_pages = set()

        with st.expander("📖 Sources"):

            for doc in st.session_state.docs:

                page = doc.metadata.get("page")

                if page is None:
                    continue

                try:
                    display_page = int(page) + 1
                except:
                    continue

                if display_page in shown_pages:
                    continue

                shown_pages.add(display_page)

                pdf_url = (
                    "https://raw.githubusercontent.com/"
                    "VishalKumar-12/HealthBot/main/"
                    "data/Medical_book.pdf"
                )

                viewer_url = (
                    "https://mozilla.github.io/pdf.js/web/viewer.html"
                    f"?file={urllib.parse.quote(pdf_url, safe='')}"
                    f"#page={display_page}"
                )

                st.markdown(
                    f"📄 **Medical_book.pdf**  \n"
                    f"Page **{display_page}**"
                )

                st.link_button(
                    f"📄 Open PDF — Page {display_page}",
                    viewer_url,
                    use_container_width=True
                )


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
