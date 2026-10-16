```python
import streamlit as st
import urllib.parse

from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="HealthBot",
    page_icon="🩺",
    layout="centered"
)


# -------------------- SIMPLE CSS --------------------

st.markdown("""
<style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
    }

    h1 {
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- TITLE --------------------

st.title("🩺 HealthBot")
st.markdown(
    '<p class="subtitle">AI Medical Assistant</p>',
    unsafe_allow_html=True
)

st.write(
    "Ask medical questions and get answers "
    "based on your medical PDF."
)


# -------------------- SESSION STATE --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" not in st.session_state:
    st.session_state.docs = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -------------------- PINECONE RETRIEVER --------------------

@st.cache_resource
def get_retriever():
    embeddings = download_embeddings()

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="healthbot-multilingual-v2-384",
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


# -------------------- GROQ LLM --------------------

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


retriever = get_retriever()
llm = get_llm()


# -------------------- PROMPT --------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (
        "human",
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer clearly and simply."
    )
])


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("⚙️ Options")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.docs = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.write("**About HealthBot**")
    st.write(
        "HealthBot answers medical questions "
        "using information retrieved from medical documents."
    )

    st.caption(
        "For education only. Not a replacement "
        "for professional medical advice."
    )


# -------------------- SUGGESTED QUESTIONS --------------------

if not st.session_state.messages:
    st.subheader("Try asking")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("What is Acne?", use_container_width=True):
            st.session_state.pending_question = "What is acne?"
            st.rerun()

        if st.button("What is Asthma?", use_container_width=True):
            st.session_state.pending_question = "What is asthma?"
            st.rerun()

    with col2:
        if st.button("What is Bronchitis?", use_container_width=True):
            st.session_state.pending_question = "What is bronchitis?"
            st.rerun()

        if st.button(
            "What is Alzheimer's disease?",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "What is Alzheimer's disease?"
            )
            st.rerun()


# -------------------- DISPLAY CHAT HISTORY --------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------- CHAT INPUT --------------------

user_input = st.chat_input("Ask your medical question...")

if not user_input and st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None


# -------------------- PROCESS QUESTION --------------------

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

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

    else:
        try:
            with st.spinner("Searching medical documents..."):
                docs = retriever.invoke(user_input)

            st.session_state.docs = docs

            context = "\n\n".join([
                (
                    f"[PDF Page "
                    f"{int(doc.metadata.get('page', 0)) + 1}]\n"
                    f"{doc.page_content}"
                )
                for doc in docs
            ])

            if not context.strip():
                context = "No relevant information was found."

            messages = prompt.format_messages(
                context=context,
                input=user_input
            )

            with st.spinner("Generating answer..."):
                response = llm.invoke(messages)
                answer = response.content

        except Exception as e:
            answer = (
                "Sorry, I could not generate an answer. "
                "Please check your API keys and connection."
            )

            st.error(f"Error: {e}")
            st.session_state.docs = []

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# -------------------- PDF SOURCES --------------------

if st.session_state.docs:

    with st.expander("📚 View Medical PDF Sources"):

        shown_pages = set()

        pdf_url = (
            "https://raw.githubusercontent.com/"
            "VishalKumar-12/HealthBot/main/"
            "data/Medical_book.pdf"
        )

        for doc in st.session_state.docs:

            page = doc.metadata.get("page")

            if page is None:
                continue

            try:
                display_page = int(page) + 1
            except (TypeError, ValueError):
                continue

            if display_page in shown_pages:
                continue

            shown_pages.add(display_page)

            st.write(f"📄 Medical_book.pdf — Page {display_page}")

            st.caption(doc.page_content[:300] + "...")

            viewer_url = (
                "https://mozilla.github.io/pdf.js/web/viewer.html"
                f"?file={urllib.parse.quote(pdf_url, safe='')}"
                f"#page={display_page}"
            )

            st.link_button(
                f"Open PDF Page {display_page}",
                viewer_url
            )


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "🩺 HealthBot | Medical information for educational purposes. "
    "Always consult a qualified healthcare professional."
)
```
