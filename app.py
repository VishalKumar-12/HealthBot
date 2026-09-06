import os
import streamlit as st

from src.Ingestion import download_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Page Configuration
# =========================================================

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
    st.error("API key is missing.")
    st.stop()


# =========================================================
# Retriever
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
# Initialize
# =========================================================

retriever = get_retriever()
llm = get_llm()


# =========================================================
# Prompt
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# =========================================================
# UI
# =========================================================

st.title("🩺 HealthBot")
st.write("AI Medical Assistant")


# =========================================================
# Chat History
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"],
            unsafe_allow_html=True
        )


# =========================================================
# User Input
# =========================================================

user_input = st.chat_input(
    "Ask your medical question..."
)


if user_input:

    # -----------------------------------------------------
    # Save User Message
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # -----------------------------------------------------
    # Greetings
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
    # Greeting Response
    # =====================================================

    if user_input.lower().strip() in greetings:

        answer = "👋 Hello! I am HealthBot. How can I help you?"

        with st.chat_message("assistant"):
            st.markdown(answer)


    # =====================================================
    # Medical Question
    # =====================================================

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching medical information..."
            ):

                try:

                    # -------------------------------------
                    # Retrieve Documents
                    # -------------------------------------

                    docs = retriever.invoke(user_input)


                    # -------------------------------------
                    # Create Context
                    # -------------------------------------

                    context = "\n\n".join(
                        f"[PDF Page "
                        f"{int(doc.metadata.get('page', 0)) + 1}]\n"
                        f"{doc.page_content}"
                        for doc in docs
                    )


                    # -------------------------------------
                    # Create Prompt
                    # -------------------------------------

                    messages = prompt.format_messages(
                        context=context,
                        input=user_input
                    )


                    # -------------------------------------
                    # Generate Answer
                    # -------------------------------------

                    response = llm.invoke(messages)

                    answer = response.content


                except Exception as e:

                    st.error(
                        "Unable to generate answer."
                    )

                    st.exception(e)

                    st.stop()


            # ---------------------------------------------
            # Display Answer
            # ---------------------------------------------

            st.markdown(
                answer,
                unsafe_allow_html=True
            )


            # =================================================
            # PDF Sources
            # =================================================

            sources = []
            seen_pages = set()


            for doc in docs:

                page = int(
                    doc.metadata.get("page", 0)
                ) + 1

                source = doc.metadata.get(
                    "source",
                    ""
                )


                if not source:
                    continue


                # ---------------------------------------------
                # Normalize Source Path
                # ---------------------------------------------

                source = source.replace(
                    "\\",
                    "/"
                )


                # ---------------------------------------------
                # Get Relative Path Inside data/
                # ---------------------------------------------

                if "data/" in source:

                    relative_path = source.split(
                        "data/",
                        1
                    )[1]

                else:

                    relative_path = os.path.basename(
                        source
                    )


                # ---------------------------------------------
                # Avoid Duplicate PDF + Page
                # ---------------------------------------------

                key = (
                    relative_path,
                    page
                )


                if key in seen_pages:
                    continue


                # ---------------------------------------------
                # GitHub PDF URL
                # ---------------------------------------------

                pdf_url = (
                    "https://raw.githubusercontent.com/"
                    "VishalKumar-12/HealthBot/main/"
                    f"data/{relative_path}#page={page}"
                )


                sources.append(
                    (
                        page,
                        relative_path,
                        pdf_url
                    )
                )


                seen_pages.add(key)


            # =================================================
            # Display Sources
            # =================================================

            if sources:

                with st.expander(
                    "📖 Sources"
                ):

                    for page, relative_path, pdf_url in sources:

                        filename = os.path.basename(
                            relative_path
                        )

                        st.markdown(
                            f'''
                            <a href="{pdf_url}"
                               target="_blank">
                                📄 Page {page} — {filename}
                            </a>
                            ''',
                            unsafe_allow_html=True
                        )


    # =====================================================
    # Save Assistant Message
    # =====================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })