from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.Ingestion import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Load Environment Variables

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# Flask App

app = Flask(__name__)


# Embedding Model

embedding = download_embeddings()


# Connect to Existing Pinecone Index

# index_name = "healthbot"
index_name = "healthbot-multilingual"

docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":6,
                #    "fetch_k":20
                   
   }
)


# Gemini LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    # model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Prompt

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# RAG Chain

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# Routes

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    # Retrieve relevant documents
    docs = retriever.invoke(msg)
    # Debug
    print("\n==========================")
    print("User Query:", msg)
    print("Retrieved Docs:", len(docs))

    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}")
        print("Source:", doc.metadata.get("source"))
        # print("Page:", doc.metadata.get("page"))
        print(doc.page_content[:300])


    

    # No documents found
    if not docs:
        return "I don't know about this because it is not available in my knowledge base."

    # Generate answer using RAG
    response = rag_chain.invoke({"input": msg})

    return response["answer"]


# Run Flask

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )