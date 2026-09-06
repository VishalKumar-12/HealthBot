from dotenv import load_dotenv
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.Ingestion import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)


# Load API key
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")


# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name = "healthbot-multilingual-384"

index_name = "healthbot-multilingual-v2-384"

# Load PDFs
documents = load_pdf_files("data")

if not documents:
    raise ValueError("No PDF found")

print("PDFs loaded:", len(documents))


# Filter documents
documents = filter_to_minimal_docs(documents)


# Create chunks
chunks = text_split(documents)

print("Chunks created:", len(chunks))


# Load embedding model
embedding = download_embeddings()

print("Embedding model loaded")


# Create Pinecone index if not exists
if not pc.has_index(index_name):

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created")


# Connect to index
index = pc.Index(index_name)


# Create vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)


# Upload chunks
batch_size = 50

for i in range(0, len(chunks), batch_size):

    batch = chunks[i:i + batch_size]

    vector_store.add_documents(batch)

    print(
        "Uploaded:",
        min(i + batch_size, len(chunks)),
        "/",
        len(chunks)
    )


print("All documents uploaded successfully!")












# from dotenv import load_dotenv
# import os

# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# from src.Ingestion import (
#     load_pdf_files,
#     filter_to_minimal_docs,
#     text_split,
#     download_embeddings,
# )

# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# pc = Pinecone(api_key=PINECONE_API_KEY)


# documents = load_pdf_files("data")
# print(f"Total PDFs Loaded: {len(documents)}")

# minimal_docs = filter_to_minimal_docs(documents)
# print(f"Minimal Documents: {len(minimal_docs)}")

# texts_chunk = text_split(minimal_docs)
# print(f"Total Chunks Created: {len(texts_chunk)}")

# embedding = download_embeddings()

# # index_name = "healthbot"
# index_name = "healthbot-multilingual"

# print(pc.describe_index(index_name))
# # index_name = "healthbot-multilingual-384"

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=1024,
#         # dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# docsearch = PineconeVectorStore.from_documents(
#     documents=texts_chunk,
#     embedding=embedding,
#     index_name=index_name
# )

# print("Pinecone index created successfully!")