import os
import urllib.parse

import streamlit as st
from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Page configuration
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
    st.error("❌ API key is missing.")
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


# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# Session state
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

    answer = ""

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)


    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
        "namaste"
    }


    # Greeting
    if user_input.lower().strip() in greetings:

        answer = "👋 Hello! I am HealthBot. How can I help you?"

        st.session_state.docs = []

        with st.chat_message("assistant"):
            st.markdown(answer)


    # Medical question
    else:

        with st.chat_message("assistant"):

            try:

                # Retrieve documents
                with st.spinner("🔎 Searching medical documents..."):

                    docs = retriever.invoke(user_input)

                st.session_state.docs = docs


                # No documents found
                if not docs:

                    answer = (
                        "Sorry, I could not find relevant "
                        "information in the medical documents."
                    )

                    st.warning(answer)


                else:

                    # Create context
                    context_parts = []

                    for doc in docs:

                        page = doc.metadata.get("page", 0)

                        try:
                            page_number = int(page) + 1
                        except Exception:
                            page_number = 1


                        pdf_name = (
                            doc.metadata.get("source")
                            or doc.metadata.get("file_name")
                            or doc.metadata.get("filename")
                            or "Medical_book.pdf"
                        )

                        pdf_name = os.path.basename(str(pdf_name))


                        context_parts.append(
                            f"[PDF: {pdf_name} | Page {page_number}]\n"
                            f"{doc.page_content}"
                        )


                    context = "\n\n".join(context_parts)


                    # Create prompt
                    messages = prompt.format_messages(
                        context=context,
                        input=user_input
                    )


                    # Generate answer
                    with st.spinner("🤖 Generating answer..."):

                        response = llm.invoke(messages)

                        answer = response.content


                    # Display answer
                    st.markdown(answer)


            except Exception as e:

                answer = (
                    "❌ Sorry, I was unable to "
                    "generate an answer."
                )

                st.error(answer)

                st.exception(e)

                st.session_state.docs = []


    # Sources
    if st.session_state.docs:

        shown_sources = set()

        with st.expander("📖 Sources", expanded=True):

            for doc in st.session_state.docs:

                page = doc.metadata.get("page")

                if page is None:
                    continue

                try:
                    display_page = int(page) + 1
                except Exception:
                    continue


                pdf_name = (
                    doc.metadata.get("source")
                    or doc.metadata.get("file_name")
                    or doc.metadata.get("filename")
                    or "Medical_book.pdf"
                )

                pdf_name = os.path.basename(str(pdf_name))


                # Avoid duplicate PDF + page
                source_key = (
                    pdf_name,
                    display_page
                )

                if source_key in shown_sources:
                    continue

                shown_sources.add(source_key)


                # GitHub PDF URL
                encoded_pdf_name = urllib.parse.quote(
                    pdf_name,
                    safe=""
                )

                pdf_url = (
                    "https://github.com/"
                    "VishalKumar-12/HealthBot/blob/main/"
                    f"data/{encoded_pdf_name}"
                    f"#page={display_page}"
                )


                # Source information
                st.markdown(
                    f"📄 **{pdf_name}**"
                )

                st.caption(
                    f"Page {display_page}"
                )


                # Open PDF
                st.link_button(
                    f"📖 Open {pdf_name} — Page {display_page}",
                    pdf_url,
                    use_container_width=True
                )


    # Save assistant message
    if answer:

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })