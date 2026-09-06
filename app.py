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
    st.error("❌ API key is missing. Please add your API keys in Streamlit Secrets.")
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
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# HEADER
# =========================================================

st.title("🩺 HealthBot")

st.caption(
    "AI Medical Assistant powered by RAG, Pinecone and Groq"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🩺 HealthBot")

    st.write(
        """
        Ask medical questions and get answers
        based on the medical PDF knowledge base.
        """
    )

    st.divider()

    st.subheader("📚 Features")

    st.write("✅ PDF-based answers")
    st.write("✅ RAG + Pinecone")
    st.write("✅ Multilingual support")
    st.write("✅ Source page references")
    st.write("✅ AI-powered responses")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
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

msg = st.chat_input("Ask your medical question...")


if msg:

    msg = msg.strip()

    if not msg:
        st.warning("Please enter a question.")
        st.stop()


    # -----------------------------------------------------
    # SHOW USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": msg
    })

    with st.chat_message("user"):
        st.markdown(msg)


    # -----------------------------------------------------
    # ASSISTANT RESPONSE
    # -----------------------------------------------------

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

            answer = "👋 Hello! I am HealthBot. How can I help you?"

            st.markdown(answer)

        else:

            # =================================================
            # SEARCH PDF
            # =================================================

            try:

                retriever = get_retriever()

                docs = retriever.invoke(msg)

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


            # =================================================
            # NO DOCUMENTS
            # =================================================

            if not docs:

                answer = "I don't have information about this topic."

                st.markdown(answer)

            else:

                # =================================================
                # CREATE CONTEXT
                # =================================================

                context = "\n\n".join(
                    f"[PDF Page {doc.metadata.get('page', 0) + 1}]\n"
                    f"{doc.page_content}"
                    for doc in docs
                )


                # =================================================
                # ASK AI
                # =================================================

                messages = prompt.format_messages(
                    context=context,
                    input=msg
                )

                try:

                    response = llm.invoke(messages)

                    answer = response.content

                except Exception as e:

                    print("Groq Error:", e)

                    answer = (
                        "Sorry, I am unable to generate "
                        "an answer right now."
                    )

                    st.error(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.stop()


                # =================================================
                # UNKNOWN ANSWER
                # =================================================

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

                    # =================================================
                    # DISPLAY ANSWER
                    # =================================================

                    st.markdown(answer)


                    # =================================================
                    # PDF SOURCES
                    # =================================================

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
                                (filename, page)
                            )


                    # Remove duplicates

                    sources = list(
                        dict.fromkeys(sources)
                    )


                    # =================================================
                    # SOURCE DISPLAY
                    # =================================================

                    if sources:

                        st.markdown("---")

                        st.markdown("### 📖 Sources")

                        for filename, page in sources:

                            pdf_url = (
                                f"/pdf/{filename}"
                                f"#page={page}"
                            )

                            st.markdown(
                                f"""
                                📄 **PDF Page {page}**  
                                """
                                f"""[Open PDF Page {page}]({pdf_url})"""
                            )


        # =================================================
        # SAVE ASSISTANT MESSAGE
        # =================================================

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })