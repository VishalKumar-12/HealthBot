from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List


# =========================================================
# Load PDF files
# =========================================================

def load_pdf_files(data):

    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


# =========================================================
# Keep only required metadata
# =========================================================

def filter_to_minimal_docs(
    docs: List[Document]
) -> List[Document]:

    minimal_docs = []

    for doc in docs:

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
            )
        )

    return minimal_docs


# =========================================================
# Split documents into chunks
# =========================================================

def text_split(minimal_docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    texts_chunk = text_splitter.split_documents(
        minimal_docs
    )

    return texts_chunk


# =========================================================
# Load Embedding Model
# =========================================================

def download_embeddings():

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings

# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document
# from typing import List


# # =========================================================
# # Extract text from PDF files
# # =========================================================

# def load_pdf_files(data):

#     loader = DirectoryLoader(
#         data,
#         glob="*.pdf",
#         loader_cls=PyPDFLoader
#     )

#     documents = loader.load()

#     return documents


# # =========================================================
# # Keep only required metadata
# # =========================================================

# def filter_to_minimal_docs(
#     docs: List[Document]
# ) -> List[Document]:

#     minimal_docs = []

#     for doc in docs:

#         minimal_docs.append(
#             Document(
#                 page_content=doc.page_content,
#                 metadata={
#                     "source": doc.metadata.get("source"),
#                     "page": doc.metadata.get("page")
#                 }
#             )
#         )

#     return minimal_docs


# # =========================================================
# # Split documents into chunks
# # =========================================================

# def text_split(minimal_docs):

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=100
#     )

#     texts_chunk = text_splitter.split_documents(
#         minimal_docs
#     )

#     return texts_chunk


# # =========================================================
# # Download / Load Embedding Model
# # =========================================================

# # def download_embeddings():

# #     # model_name = "sentence-transformers/all-MiniLM-L6-v2"
# #     model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# #     embeddings = HuggingFaceEmbeddings(
# #         model_name=model_name,
# #         model_kwargs={
# #             "device": "cpu"
# #         },
# #         encode_kwargs={
# #             "normalize_embeddings": True
# #         }
# #     )

# #     return embeddings

# # def download_embeddings():

# #     model_name = "BAAI/bge-m3"

# #     embeddings = HuggingFaceEmbeddings(
# #         model_name=model_name,
# #         model_kwargs={
# #             "device": "cpu"
# #         },
# #         encode_kwargs={
# #             "normalize_embeddings": True
# #         }
# #     )

# #     return embeddings

# def download_embeddings():

#     model_name = "sentence-transformers/all-MiniLM-L6-v2"

#     embeddings = HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs={
#             "device": "cpu"
#         },
#         encode_kwargs={
#             "normalize_embeddings": True
#         }
#     )

#     return embeddings