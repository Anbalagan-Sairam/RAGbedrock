import os
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# -----------------------------
# Paths
# -----------------------------
VECTORSTORE_PATH = Path("vectorstore").resolve()
DATA_FOLDER = Path("data").resolve()

def load_documents(data_folder=DATA_FOLDER):
    """Load all TXT and PDF files from the folder."""
    texts = []
    files_found = [f for f in os.listdir(data_folder) if f.endswith((".txt", ".pdf"))]
    print(f"[INFO] Found {len(files_found)} documents in '{data_folder}': {files_found}")

    for file in files_found:
        filepath = data_folder / file
        if file.endswith(".txt"):
            print(f"[INFO] Loading TXT file: {file}")
            with open(filepath, "r") as f:
                content = f.read()
                texts.append(content)
        elif file.endswith(".pdf"):
            print(f"[INFO] Loading PDF file: {file}")
            reader = PdfReader(filepath)
            pdf_text = ""
            for i, page in enumerate(reader.pages, 1):
                page_content = page.extract_text() or ""
                pdf_text += page_content + "\n"
            if pdf_text.strip():
                texts.append(pdf_text)
    return texts

def split_documents(texts):
    """Split texts into chunks for embedding."""
    docs = [Document(page_content=t) for t in texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split {len(texts)} documents into {len(chunks)} chunks")
    return chunks

def build_vectorstore(embedding_model_id, aws_region, data_folder=DATA_FOLDER):
    """Always rebuild FAISS vectorstore from documents (safe, no pickle loading)."""
    print("=== TrailblazeAI Vectorstore Builder Started ===")

    # Ensure vectorstore folder exists
    if VECTORSTORE_PATH.exists():
        import shutil
        print("[INFO] Removing old vectorstore folder...")
        shutil.rmtree(VECTORSTORE_PATH)
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    # Load and split documents
    texts = load_documents(data_folder)
    if not texts:
        print("[WARN] No documents found. Exiting.")
        return

    chunks = split_documents(texts)

    # Initialize embeddings
    print(f"[INFO] Initializing BedrockEmbeddings with model '{embedding_model_id}' in region '{aws_region}'...")
    embedding_function = BedrockEmbeddings(
        model_id=embedding_model_id,
        region_name=aws_region
    )

    # Always rebuild FAISS index
    print("[INFO] Building FAISS vectorstore from document chunks...")
    vectorstore = FAISS.from_texts([c.page_content for c in chunks], embedding_function)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("[INFO] Vectorstore saved locally.")

    print("=== Vectorstore Builder Completed ===")
    return vectorstore

# -----------------------------
# Standalone execution
# -----------------------------
if __name__ == "__main__":
    load_dotenv()  # Load BEDROCK_EMBED_MODEL, AWS_REGION

    build_vectorstore(
        embedding_model_id=os.environ.get("BEDROCK_EMBED_MODEL"),
        aws_region=os.environ.get("AWS_REGION"),
        data_folder=DATA_FOLDER
    )