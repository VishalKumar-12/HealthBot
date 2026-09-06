import os
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


# Load models

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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    /* Source links */

    .source-link {
        display: block;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 8px;
        text-decoration: none !important;
        border: 1px solid #ddd;
    }

    .source-link:hover {
        background-color: #f5f5f5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 HealthBot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Medical Assistant</div>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


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
# USER QUESTION
# =========================================================

if user_input:

    user_input = user_input.strip()

    if not user_input:
        st.stop()


    # -----------------------------------------------------
    # Save user message
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)


    # =====================================================
    # GREETING
    # =====================================================

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

        answer = (
            "👋 Hello! I am HealthBot. "
            "How can I help you?"
        )

        with st.chat_message("assistant"):

            st.markdown(answer)


        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.stop()


    # =====================================================
    # MEDICAL QUESTION
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching medical information..."
        ):

            try:

                # -------------------------------------------------
                # Retrieve documents from Pinecone
                # -------------------------------------------------

                docs = retriever.invoke(user_input)


                # -------------------------------------------------
                # No documents found
                # -------------------------------------------------

                if not docs:

                    answer = (
                        "I don't have information "
                        "about this topic in my medical sources."
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

                context_parts = []

                for doc in docs:

                    page_number = (
                        int(
                            doc.metadata.get(
                                "page",
                                0
                            )
                        ) + 1
                    )

                    context_parts.append(
                        f"[PDF Page {page_number}]\n"
                        f"{doc.page_content}"
                    )


                context = "\n\n".join(
                    context_parts
                )


                # -------------------------------------------------
                # Create prompt
                # -------------------------------------------------

                messages = prompt.format_messages(
                    context=context,
                    input=user_input
                )


                # -------------------------------------------------
                # Generate answer
                # -------------------------------------------------

                with st.spinner(
                    "Generating answer..."
                ):

                    response = llm.invoke(
                        messages
                    )

                    answer = response.content


            except Exception as e:

                st.error(
                    "Unable to generate answer."
                )

                st.exception(e)

                st.stop()


        # =====================================================
        # DISPLAY ANSWER
        # =====================================================

        st.markdown(
            answer,
            unsafe_allow_html=True
        )


        # =====================================================
        # PDF SOURCES
        # =====================================================

        sources = []

        seen_pages = set()


        for doc in docs:

            try:

                page = (
                    int(
                        doc.metadata.get(
                            "page",
                            0
                        )
                    ) + 1
                )

            except Exception:

                continue


            # Avoid duplicate pages

            if page in seen_pages:

                continue


            seen_pages.add(page)


            # -------------------------------------------------
            # GitHub PDF URL
            #
            # #page=N tells browser PDF viewer
            # which page to open.
            # -------------------------------------------------

            pdf_url = (
                "https://github.com/"
                "VishalKumar-12/HealthBot/"
                "raw/refs/heads/main/"
                "data/Medical_book.pdf"
                f"#page={page}"
            )


            sources.append(
                (page, pdf_url)
            )


        # =====================================================
        # SHOW SOURCES
        # =====================================================

        if sources:

            with st.expander(
                "📖 Sources",
                expanded=False
            ):

                for page, pdf_url in sources:

                    st.markdown(
                        f"""
                        <a
                            class="source-link"
                            href="{pdf_url}"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            📄 Page {page} — Medical_book.pdf
                        </a>
                        """,
                        unsafe_allow_html=True
                    )


    # =========================================================
    # SAVE ASSISTANT MESSAGE
    # =========================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })



# import os
# import streamlit as st

# from src.Ingestion import download_embeddings
# from src.prompt import system_prompt
# from langchain_pinecone import PineconeVectorStore
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate


# st.set_page_config(
#     page_title="HealthBot",
#     page_icon="🩺",
#     layout="wide"
# )


# # API Keys
# try:
#     PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
#     GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# except Exception:
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not PINECONE_API_KEY or not GROQ_API_KEY:
#     st.error("API key is missing.")
#     st.stop()


# # Retriever
# @st.cache_resource
# def get_retriever():

#     embedding = download_embeddings()

#     vector_store = PineconeVectorStore(
#         index_name="healthbot-multilingual-v2-384",
#         embedding=embedding,
#         pinecone_api_key=PINECONE_API_KEY
#     )

#     return vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 4}
#     )


# # LLM
# @st.cache_resource
# def get_llm():

#     return ChatGroq(
#         model="openai/gpt-oss-20b",
#         api_key=GROQ_API_KEY,
#         temperature=0
#     )


# retriever = get_retriever()
# llm = get_llm()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}")
# ])


# # UI
# st.title("🩺 HealthBot")
# st.write("AI Medical Assistant")


# if "messages" not in st.session_state:
#     st.session_state.messages = []


# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(
#             message["content"],
#             unsafe_allow_html=True
#         )


# user_input = st.chat_input("Ask your medical question...")


# if user_input:

#     st.session_state.messages.append({
#         "role": "user",
#         "content": user_input
#     })

#     with st.chat_message("user"):
#         st.markdown(user_input)


#     # Greeting
#     greetings = [
#         "hi", "hello", "hey", "hii", "hiii",
#         "good morning", "good afternoon",
#         "good evening", "namaste"
#     ]

#     if user_input.lower().strip() in greetings:

#         answer = "👋 Hello! I am HealthBot. How can I help you?"

#     else:

#         with st.chat_message("assistant"):

#             with st.spinner("Searching medical information..."):

#                 try:
#                     docs = retriever.invoke(user_input)

#                     context = "\n\n".join(
#                         f"[PDF Page {doc.metadata.get('page', 0) + 1}]\n"
#                         f"{doc.page_content}"
#                         for doc in docs
#                     )

#                     messages = prompt.format_messages(
#                         context=context,
#                         input=user_input
#                     )

#                     response = llm.invoke(messages)
#                     answer = response.content

#                 except Exception as e:

#                     st.error("Unable to generate answer.")
#                     st.exception(e)
#                     st.stop()


#             st.markdown(
#                 answer,
#                 unsafe_allow_html=True
#             )


#             # Sources
#             sources = []
#             seen_pages = set()

#             for doc in docs:

#                 page = int(
#                     doc.metadata.get("page", 0)
#                 ) + 1

#                 if page not in seen_pages:

#                     pdf_url = (
#                         "https://raw.githubusercontent.com/"
#                         "VishalKumar-12/HealthBot/main/"
#                         f"data/Medical_book.pdf#page={page}"
#                     )

#                     sources.append(
#                         (page, pdf_url)
#                     )

#                     seen_pages.add(page)


#             if sources:

#                 with st.expander("📖 Sources"):

#                     for page, pdf_url in sources:

#                         st.markdown(
#                             f'<a href="{pdf_url}" target="_blank">'
#                             f'📄 Page {page} — Medical_book.pdf'
#                             f'</a>',
#                             unsafe_allow_html=True
#                         )


#     if user_input.lower().strip() in greetings:

#         with st.chat_message("assistant"):
#             st.markdown(answer)

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": answer
#     })