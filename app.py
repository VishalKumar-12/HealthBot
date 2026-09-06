from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing")


app = Flask(__name__)

INDEX_NAME = "healthbot-multilingual-384"


retriever = None


def get_retriever():
    global retriever

    if retriever is not None:
        return retriever

    print("Loading embedding model...")

    embedding = download_embeddings()

    print("Connecting to Pinecone...")

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    print("Retriever ready.")

    return retriever


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question."

    # Retrieve relevant documents
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(msg)

    except Exception as e:
        print("Pinecone/Embedding Error:", e)

        return (
            "Sorry, the medical search service is "
            "temporarily unavailable."
        )

    if not docs:
        return (
            "I don't know about this because the information "
            "is not available in my medical PDF."
        )

    # Combine retrieved document content
    pdf_context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Create prompt
    final_messages = prompt.format_messages(
        context=pdf_context,
        input=msg
    )

    # Generate answer
    try:
        response = llm.invoke(final_messages)

    except Exception as e:
        print("Groq Error:", e)

        return (
            "Sorry, I am unable to generate an answer "
            "right now. Please try again."
        )

    return response.content


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )