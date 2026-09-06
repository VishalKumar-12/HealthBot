import os
import base64
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
# PDF PATH
# =========================================================

PDF_FOLDER = "data"


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
        search_kwargs={"k": 4}
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


llm = get_llm()


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# =========================================================
# PDF FILE
# =========================================================

def get_pdf_path(source):

    if not source:
        return None

    # Convert Windows path to normal path
    source = source.replace("\\", "/")

    # Get only filename
    filename = os.path.basename(source)

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    if os.path.exists(pdf_path):
        return pdf_path

    return None


# =========================================================
# PDF PAGE LINK
# =========================================================

def create_pdf_link(source, page):

    pdf_path = get_pdf_path(source)

    if not pdf_path:
        return None

    try:

        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        base64_pdf = base64.b64encode(
            pdf_bytes
        ).decode("utf-8")

        pdf_url = (
            "data:application/pdf;base64,"
            + base64_pdf
            + f"#page={page}"
        )

        return pdf_url

    except Exception as e:

        print("PDF Error:", e)

        return None


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🩺 HealthBot")

    st.write(
        "AI Medical Assistant powered by "
        "RAG, Pinecone and Groq."
    )

    st.divider()

    st.subheader("📚 Features")

    st.write("✅ PDF-based answers")
    st.write("✅ RAG + Pinecone")
    st.write("✅ Multilingual support")
    st.write("✅ PDF page references")
    st.write("✅ AI-powered responses")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.title("🩺 HealthBot")

st.caption(
    "AI Medical Assistant • RAG + Pinecone + Groq"
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
# USER INPUT
# =========================================================

msg = st.chat_input(
    "Ask your medical question..."
)


# =========================================================
# CHAT
# =========================================================

if msg:

    msg = msg.strip()

    if not msg:

        st.warning("Please enter a question.")

        st.stop()


    # =====================================================
    # USER MESSAGE
    # =====================================================

    st.session_state.messages.append({
        "role": "user",
        "content": msg
    })

    with st.chat_message("user"):

        st.markdown(msg)


    # =====================================================
    # ASSISTANT MESSAGE
    # =====================================================

    with st.chat_message("assistant"):

        # =================================================
        # GREETING
        # =================================================

        if msg.lower() in [
            "hi",
            "hello",
            "hey",
            "hii",
            "hiii",
            "good morning",
            "good afternoon",
            "good evening",
            "namaste"
        ]:

            answer = (
                "👋 Hello! I am HealthBot. "
                "How can I help you?"
            )

            st.markdown(answer)


        else:

            # =============================================
            # SEARCH PDF
            # =============================================

            try:

                with st.spinner(
                    "🔍 Searching medical information..."
                ):

                    docs = get_retriever().invoke(msg)

            except Exception as e:

                print("Pinecone Error:", e)

                answer = (
                    "Sorry, the medical search service "
                    "is temporarily unavailable."
                )

                st.error(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.stop()


            # =============================================
            # NO DOCUMENTS
            # =============================================

            if not docs:

                answer = (
                    "I don't have information "
                    "about this topic."
                )

                st.markdown(answer)


            else:

                # =========================================
                # CREATE CONTEXT
                # =========================================

                context = "\n\n".join(
                    f"[PDF Page "
                    f"{doc.metadata.get('page', 0) + 1}]\n"
                    f"{doc.page_content}"
                    for doc in docs
                )


                # =========================================
                # ASK AI
                # =========================================

                messages = prompt.format_messages(
                    context=context,
                    input=msg
                )

                try:

                    with st.spinner(
                        "🤖 Generating answer..."
                    ):

                        response = llm.invoke(messages)

                    answer = response.content

                except Exception as e:

                    print("Groq Error:", e)

                    answer = (
                        "Sorry, I am unable to "
                        "generate an answer right now."
                    )

                    st.error(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.stop()


                # =========================================
                # UNKNOWN ANSWER
                # =========================================

                unknown = [
                    "don't know",
                    "do not know",
                    "not available",
                    "not found",
                    "not mentioned"
                ]


                if any(
                    word in answer.lower()
                    for word in unknown
                ):

                    st.markdown(answer)


                else:

                    # =====================================
                    # DISPLAY ANSWER
                    # =====================================

                    st.markdown(answer)


                    # =====================================
                    # PDF SOURCES
                    # =====================================

                    sources = []

                    for doc in docs:

                        page = int(
                            doc.metadata.get(
                                "page",
                                0
                            )
                        ) + 1

                        source = doc.metadata.get(
                            "source",
                            ""
                        )

                        if source:

                            # Normalize path
                            source = source.replace(
                                "\\",
                                "/"
                            )

                            filename = os.path.basename(
                                source
                            )

                            sources.append(
                                (
                                    filename,
                                    source,
                                    page
                                )
                            )


                    # Remove duplicate sources

                    unique_sources = []

                    seen = set()

                    for filename, source, page in sources:

                        key = (
                            filename,
                            page
                        )

                        if key not in seen:

                            seen.add(key)

                            unique_sources.append(
                                (
                                    filename,
                                    source,
                                    page
                                )
                            )


                    # =====================================
                    # SHOW SOURCES
                    # =====================================

                    if unique_sources:

                        st.markdown("---")

                        st.markdown(
                            "### 📖 Sources"
                        )

                        for (
                            filename,
                            source,
                            page
                        ) in unique_sources:

                            pdf_link = create_pdf_link(
                                source,
                                page
                            )

                            if pdf_link:

                                st.markdown(
                                    f"""
                                    <a
                                        href="{pdf_link}"
                                        target="_blank"
                                        style="
                                            text-decoration:none;
                                            font-weight:500;
                                        "
                                    >
                                        📄 PDF Page {page}
                                    </a>
                                    """,
                                    unsafe_allow_html=True
                                )

                            else:

                                st.write(
                                    f"📄 PDF Page {page}"
                                )


    # =====================================================
    # SAVE ASSISTANT RESPONSE
    # =====================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

