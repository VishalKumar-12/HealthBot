from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.Ingestion import download_embeddings
from src.prompt import system_prompt
from src.api.medlineplus import search_medlineplus
from src.api.context import format_medlineplus

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)


# =========================================================
# Embedding Model
# =========================================================

embedding = download_embeddings()


# =========================================================
# Connect to Existing Pinecone Index
# =========================================================

# index_name = "healthbot-multilingual"
index_name = "healthbot-multilingual-v2"

docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding,
)


# =========================================================
# Retriever
# =========================================================

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6
    }
)


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
# Routes
# =========================================================

@app.route("/")
def index():
    return render_template("chat.html")


# =========================================================
# Chat Route
# =========================================================

@app.route("/get", methods=["POST"])
def chat():

    # -----------------------------------------------------
    # Get user question
    # -----------------------------------------------------

    msg = request.form["msg"]

    print("\n======================================")
    print("USER QUESTION:")
    print(msg)
    print("======================================")


    # =====================================================
    # 1. Retrieve information from PDF / Pinecone
    # =====================================================

    docs = retriever.invoke(msg)

    print("\nPDF RESULTS:", len(docs))


    for i, doc in enumerate(docs):

        print(f"\nPDF Document {i + 1}")

        print(
            "Source:",
            doc.metadata.get("source")
        )

        print(
            doc.page_content[:300]
        )


    # =====================================================
    # 2. Fetch information from MedlinePlus API
    # =====================================================

    api_results = search_medlineplus(msg)

    print("\nMEDLINEPLUS RESULTS:", len(api_results))


    # =====================================================
    # 3. Convert MedlinePlus results into text
    # =====================================================

    api_context = format_medlineplus(
        api_results
    )


    # =====================================================
    # 4. Convert PDF documents into text
    # =====================================================

    if docs:

        pdf_context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

    else:

        pdf_context = "No relevant information found in the PDF."


    # =====================================================
    # 5. Combine PDF + MedlinePlus
    # =====================================================

    combined_context = f"""

==============================
PDF MEDICAL INFORMATION
==============================

{pdf_context}


==============================
MEDLINEPLUS MEDICAL INFORMATION
==============================

{api_context}

"""


    # =====================================================
    # Debug: Show combined context
    # =====================================================

    print("\n======================================")
    print("COMBINED CONTEXT CREATED")
    print("======================================")

    print(
        combined_context[:2000]
    )


    # =====================================================
    # 6. Check if both sources are empty
    # =====================================================

    if not docs and not api_results:

        return (
            "I don't know about this because it is not "
            "available in my medical sources."
        )


    # =====================================================
    # 7. Create final prompt
    # =====================================================

    final_messages = prompt.format_messages(
        context=combined_context,
        input=msg
    )


    # =====================================================
    # 8. Send PDF + API context to LLM
    # =====================================================

    response = llm.invoke(
        final_messages
    )


    # =====================================================
    # 9. Return final answer
    # =====================================================

    return response.content


# =========================================================
# Run Flask
# =========================================================

# if __name__ == "__main__":

#     app.run(
#         host="127.0.0.1",
#         port=5000,
#         debug=True
#     )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
