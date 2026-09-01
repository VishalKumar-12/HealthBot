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

<<<<<<< HEAD
=======
embedding = download_embeddings()


# =========================================================
# Connect to Existing Pinecone Index
# =========================================================

>>>>>>> e21b5b2 (Update medical chatbot)
index_name = "healthbot-multilingual-384"

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

    if retriever is None:

        print("\n======================================")
        print("Loading embedding model...")
        print("======================================")

        embedding = download_embeddings()

        print("Embedding model loaded.")

        print("\n======================================")
        print("Connecting to Pinecone...")
        print("======================================")

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

        docs = []


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
            doc.page_content[:300]
        )


    # =====================================================
<<<<<<< HEAD
    # 3. Fetch Information from MedlinePlus
    # =====================================================

    try:

        api_results = search_medlineplus(msg)

    except Exception as e:

        print("\nMEDLINEPLUS API ERROR:")
        print(e)

        api_results = []


    print(
        "\nMEDLINEPLUS RESULTS:",
        len(api_results)
    )


    # =====================================================
    # 4. Convert MedlinePlus Results to Text
    # =====================================================

    try:

        api_context = format_medlineplus(
            api_results
        )

    except Exception as e:

        print("\nMEDLINEPLUS FORMAT ERROR:")
        print(e)

        api_context = "No MedlinePlus information available."


    # =====================================================
    # 5. Convert PDF Documents to Text
=======
    # 2. Convert PDF documents into text
>>>>>>> e21b5b2 (Update medical chatbot)
    # =====================================================

    if docs:

        pdf_context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

    else:

        pdf_context = (
            "No relevant information found "
            "in the medical PDF."
        )


    # =====================================================
<<<<<<< HEAD
    # 6. Combine PDF + MedlinePlus
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
    # Debug Context
    # =====================================================

    print("\n======================================")
    print("COMBINED CONTEXT CREATED")
    print("======================================")

    print(
        combined_context[:2000]
    )


    # =====================================================
    # 7. Check if Both Sources Are Empty
    # =====================================================

    if not docs and not api_results:
=======
    # 3. Check if PDF information is available
    # =====================================================

    if not docs:
>>>>>>> e21b5b2 (Update medical chatbot)

        return (
            "I don't know about this because it is not "
            "available in my medical PDF."
        )


    # =====================================================
<<<<<<< HEAD
    # 8. Create Final Prompt
=======
    # 4. Create final prompt
>>>>>>> e21b5b2 (Update medical chatbot)
    # =====================================================

    final_messages = prompt.format_messages(
        context=pdf_context,
        input=msg
    )


    # =====================================================
<<<<<<< HEAD
    # 9. Send Context to Groq
=======
    # 5. Send PDF context to LLM
>>>>>>> e21b5b2 (Update medical chatbot)
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
<<<<<<< HEAD
    # 10. Return Final Answer
=======
    # 6. Return final answer
>>>>>>> e21b5b2 (Update medical chatbot)
    # =====================================================

    return response.content


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

<<<<<<< HEAD
    port = int(
        os.environ.get("PORT", 5000)
    )
=======
    port = int(os.environ.get("PORT", 5000))
>>>>>>> e21b5b2 (Update medical chatbot)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
