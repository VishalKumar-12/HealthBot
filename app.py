
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing")


# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)


# =========================================================
# Pinecone Configuration
# =========================================================

index_name = "healthbot-multilingual-384"

# Lazy loading
embedding = None
docsearch = None
retriever = None


# =========================================================
# Lazy Load Embedding Model + Pinecone
# =========================================================

def get_retriever():

    global embedding
    global docsearch
    global retriever

    # If retriever already exists, reuse it
    if retriever is not None:
        return retriever

    print("\n======================================")
    print("Loading embedding model...")
    print("======================================")

    try:
        embedding = download_embeddings()

        print("Embedding model loaded.")

    except Exception as e:

        print("\nEMBEDDING ERROR:")
        print(e)

        raise


    print("\n======================================")
    print("Connecting to Pinecone...")
    print("======================================")

    try:

        docsearch = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding
        )

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4
            }
        )

        print("Pinecone retriever ready.")

    except Exception as e:

        print("\nPINECONE CONNECTION ERROR:")
        print(e)

        raise


    return retriever


# =========================================================
# Groq LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0
)


# =========================================================
# Prompt
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# =========================================================
# Home Route
# =========================================================

@app.route("/")
def index():

    return render_template("chat.html")


# =========================================================
# Chat Route
# =========================================================

@app.route("/get", methods=["POST"])
def chat():

    # =====================================================
    # Get User Question
    # =====================================================

    msg = request.form.get("msg", "").strip()

    if not msg:

        return "Please enter a question."


    print("\n======================================")
    print("USER QUESTION:")
    print(msg)
    print("======================================")


    # =====================================================
    # 1. Get Pinecone Retriever
    # =====================================================

    try:

        retriever_instance = get_retriever()

    except Exception as e:

        print("\nPINECONE / EMBEDDING ERROR:")
        print(e)

        return (
            "Sorry, the medical search service is "
            "temporarily unavailable."
        )


    # =====================================================
    # 2. Retrieve Information from Pinecone
    # =====================================================

    try:

        docs = retriever_instance.invoke(msg)

    except Exception as e:

        print("\nPINECONE SEARCH ERROR:")
        print(e)

        return (
            "Sorry, I am unable to search the medical "
            "information right now."
        )


    print("\nPDF RESULTS:", len(docs))


    # =====================================================
    # Print Retrieved Documents
    # =====================================================

    for i, doc in enumerate(docs):

        print(f"\nPDF Document {i + 1}")

        print(
            "Source:",
            doc.metadata.get("source")
        )

        print(
            "Page:",
            doc.metadata.get("page")
        )

        print(
            "Content:",
            doc.page_content[:300]
        )


    # =====================================================
    # 3. Convert PDF Documents into Text
    # =====================================================

    if docs:

        pdf_context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

    else:

        pdf_context = (
            "No relevant information was found "
            "in the medical PDF."
        )


    # =====================================================
    # 4. Check if PDF Information is Available
    # =====================================================

    if not docs:

        return (
            "I don't know about this because the information "
            "is not available in my medical PDF."
        )


    # =====================================================
    # 5. Create Final Prompt
    # =====================================================

    final_messages = prompt.format_messages(
        context=pdf_context,
        input=msg
    )


    # =====================================================
    # 6. Send Context to Groq
    # =====================================================

    try:

        response = llm.invoke(
            final_messages
        )

    except Exception as e:

        print("\nGROQ ERROR:")
        print(e)

        return (
            "Sorry, I am unable to generate an answer "
            "right now. Please try again."
        )


    # =====================================================
    # 7. Return Final Answer
    # =====================================================

    return response.content


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

