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


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env file")


# ============================================================
# Initialize Pinecone
# ============================================================

pc = Pinecone(api_key=PINECONE_API_KEY)


# ============================================================
# 1. Load PDF files
# ============================================================

print("\n========================================")
print("1. Loading PDF files...")
print("========================================")

documents = load_pdf_files("data")

print(f"Total PDFs Loaded: {len(documents)}")

if not documents:
    raise ValueError(
        "No PDF files found inside the 'data' folder."
    )


# ============================================================
# 2. Filter documents
# ============================================================

print("\n========================================")
print("2. Filtering documents...")
print("========================================")

minimal_docs = filter_to_minimal_docs(documents)

print(f"Minimal Documents: {len(minimal_docs)}")


# ============================================================
# 3. Create chunks
# ============================================================

print("\n========================================")
print("3. Creating text chunks...")
print("========================================")

texts_chunk = text_split(minimal_docs)

print(f"Total Chunks Created: {len(texts_chunk)}")

if not texts_chunk:
    raise ValueError(
        "No chunks were created from the PDFs."
    )


# ============================================================
# 4. Load embedding model
# ============================================================

print("\n========================================")
print("4. Loading embedding model...")
print("========================================")

embedding = download_embeddings()

print("Embedding model loaded successfully.")
print("Model: all-MiniLM-L6-v2")
print("Dimension: 384")


# ============================================================
# 5. Pinecone Index
# ============================================================

index_name = "healthbot-multilingual-384"

print("\n========================================")
print("5. Checking Pinecone index...")
print("========================================")

if not pc.has_index(index_name):

    print(
        f"Creating Pinecone index: {index_name}"
    )

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created successfully!")

else:

    print(
        f"Pinecone index '{index_name}' already exists."
    )


# ============================================================
# 6. Connect to Pinecone index
# ============================================================

print("\n========================================")
print("6. Connecting to Pinecone...")
print("========================================")

index = pc.Index(index_name)

print("Pinecone index connected.")

print("\nCurrent Index Stats:")
print(index.describe_index_stats())


# ============================================================
# 7. Create Vector Store
# ============================================================

print("\n========================================")
print("7. Creating Vector Store...")
print("========================================")

docsearch = PineconeVectorStore(
    index=index,
    embedding=embedding
)

print("Vector store created successfully.")


# ============================================================
# 8. Upload documents in batches
# ============================================================

print("\n========================================")
print("8. Uploading documents...")
print("========================================")

BATCH_SIZE = 50

total_chunks = len(texts_chunk)

print(f"Total chunks: {total_chunks}")
print(f"Batch size: {BATCH_SIZE}")


for i in range(0, total_chunks, BATCH_SIZE):

    batch = texts_chunk[i:i + BATCH_SIZE]

    start = i + 1
    end = min(
        i + BATCH_SIZE,
        total_chunks
    )

    print(
        f"\nUploading chunks "
        f"{start} - {end} "
        f"of {total_chunks}"
    )

    try:

        docsearch.add_documents(batch)

        print(
            f"Successfully uploaded "
            f"chunks {start} - {end}"
        )

    except Exception as e:

        print(
            f"\nERROR uploading "
            f"chunks {start} - {end}"
        )

        print(e)

        raise


# ============================================================
# 9. Final Statistics
# ============================================================

print("\n========================================")
print("9. Final Pinecone Statistics")
print("========================================")

final_stats = index.describe_index_stats()

print(final_stats)


# ============================================================
# 10. Done
# ============================================================

print("\n========================================")
print("ALL DOCUMENTS UPLOADED SUCCESSFULLY!")
print("========================================")

print(f"Index Name  : {index_name}")
print("Embedding   : all-MiniLM-L6-v2")
print("Dimension   : 384")
print(f"Total Chunks: {total_chunks}")

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