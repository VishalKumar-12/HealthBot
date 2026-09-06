from flask import Flask, render_template, request, send_from_directory
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

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("API key is missing")


app = Flask(__name__)

retriever = None


def get_retriever():
    global retriever

    if retriever is None:
        embedding = download_embeddings()

        vector_store = PineconeVectorStore(
            index_name="healthbot-multilingual-v2-384",
            embedding=embedding
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

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


@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    return send_from_directory("data", filename)


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question."

    # Greeting
    if msg.lower() in [
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon",
        "good evening", "namaste"
    ]:
        return "👋 Hello! I am HealthBot. How can I help you?"

    # Search PDF
    try:
        docs = get_retriever().invoke(msg)
    except Exception as e:
        print("Pinecone Error:", e)
        return "Sorry, the medical search service is temporarily unavailable."

    if not docs:
        return "I don't have information about this topic."

    # Create context
    context = "\n\n".join(
        f"[PDF Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in docs
    )

    # Ask AI
    messages = prompt.format_messages(
        context=context,
        input=msg
    )

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print("Groq Error:", e)
        return "Sorry, I am unable to generate an answer right now."

    answer = response.content

    # Unknown answer
    unknown = [
        "don't know",
        "do not know",
        "not available",
        "not found",
        "not mentioned"
    ]

    if any(word in answer.lower() for word in unknown):
        return answer

    # PDF pages
    sources = []

    for doc in docs:

        page = int(doc.metadata.get("page", 0)) + 1
        source = doc.metadata.get("source", "")

        if source:
            filename = os.path.basename(source)

            sources.append(
                f"- 📄 <a href='/pdf/{filename}#page={page}' "
                f"target='_blank'>PDF Page {page}</a>"
            )

    sources = list(dict.fromkeys(sources))

    if sources:
        answer += "\n\n### 📖 Sources\n" + "\n".join(sources)

    return answer


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )