from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# Extract text from PDF files

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls= PyPDFLoader
    )

    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
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

def text_split(mininal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
    )
    texts_chunk = text_splitter.split_documents(mininal_docs)
    return texts_chunk


def download_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

# def download_embeddings():
#     """
#       Download and return the HuggingFace embeddings model.
#     """
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     embeddings = HuggingFaceEmbeddings(
#         model_name = model_name
#     )
#     embedding = download_embeddings()

