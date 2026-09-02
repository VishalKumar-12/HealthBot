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


# ==========================================
# 1. Load PDF files
# ==========================================

documents = load_pdf_files("data")
print(f"Total PDFs Loaded: {len(documents)}")


# ==========================================
# 2. Filter documents
# ==========================================

minimal_docs = filter_to_minimal_docs(documents)
print(f"Minimal Documents: {len(minimal_docs)}")


# ==========================================
# 3. Create chunks
# ==========================================

texts_chunk = text_split(minimal_docs)
print(f"Total Chunks Created: {len(texts_chunk)}")


# ==========================================
# 4. Load embedding model
# ==========================================

embedding = download_embeddings()


# ==========================================
# 5. Pinecone Index
# ==========================================

index_name = "healthbot-multilingual"

if not pc.has_index(index_name):

    print("Creating Pinecone index...")

    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created successfully!")

else:

    print("Pinecone index already exists.")


# ==========================================
# 6. Connect to Pinecone index
# ==========================================

index = pc.Index(index_name)

print(index.describe_index_stats())


# ==========================================
# 7. Create Vector Store
# ==========================================

docsearch = PineconeVectorStore(
    index=index,
    embedding=embedding
)


# ==========================================
# 8. Upload documents in small batches
# ==========================================

BATCH_SIZE = 50

total_chunks = len(texts_chunk)

for i in range(0, total_chunks, BATCH_SIZE):

    batch = texts_chunk[i:i + BATCH_SIZE]

    start = i + 1
    end = min(i + BATCH_SIZE, total_chunks)

    print(f"Uploading chunks {start} - {end} of {total_chunks}")

    docsearch.add_documents(batch)


# ==========================================
# 9. Done
# ==========================================

print("========================================")
print("All documents uploaded successfully!")
print("========================================")



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