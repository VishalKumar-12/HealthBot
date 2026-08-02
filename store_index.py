from dotenv import load_dotenv
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.Ingestion import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings,
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

documents = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(documents)
texts_chunk = text_split(minimal_docs)

embedding = download_embeddings()

# index_name = "healthbot"
index_name = "healthbot-multilingual"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        # dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)

print("Pinecone index created successfully!")