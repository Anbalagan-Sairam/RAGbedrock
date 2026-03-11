#  app/vector_manager.py
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

VECTORSTORE_PATH = "vectorstore"

def load_documents(data_folder="data"):
    """Load all TXT and PDF files from the folder."""
    texts = []
    files_found = os.listdir(data_folder)
    print(f"[INFO] Found {len(files_found)} files in '{data_folder}': {files_found}")

    for file in files_found:
        filepath = os.path.join(data_folder, file)
        if file.endswith(".txt"):
            print(f"[INFO] Loading TXT file: {file}")
            with open(filepath, "r") as f:
                content = f.read()
                texts.append(content)
                print(f"  -> Loaded {len(content)} characters from {file}")
        elif file.endswith(".pdf"):
            print(f"[INFO] Loading PDF file: {file}")
            reader = PdfReader(filepath)
            pdf_text = ""
            for i, page in enumerate(reader.pages, 1):
                page_content = page.extract_text()
                if page_content:
                    pdf_text += page_content + "\n"
                print(f"  -> Page {i}: {len(page_content) if page_content else 0} characters extracted")
            if pdf_text.strip():
                texts.append(pdf_text)
                print(f"  -> Total {len(pdf_text)} characters added from {file}")
    return texts

def split_documents(texts):
    """Split texts into chunks for embedding."""
    docs = [Document(page_content=t) for t in texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split {len(texts)} documents into {len(chunks)} chunks (chunk_size=500, overlap=50)")
    return chunks

def build_vectorstore(embedding_model_id, aws_region, data_folder="data"):
    """Build or load the FAISS vectorstore with AWS Bedrock embeddings."""
    print("=== TrailblazeAI Vectorstore Builder Started ===")

    texts = load_documents(data_folder)
    if not texts:
        print("[WARN] No documents found. Exiting.")
        return

    chunks = split_documents(texts)

    print(f"[INFO] Initializing BedrockEmbeddings with model '{embedding_model_id}' in region '{aws_region}'...")
    embedding_function = BedrockEmbeddings(
        model_id=embedding_model_id,
        region_name=aws_region
    )

    if os.path.exists(VECTORSTORE_PATH):
        print("[INFO] FAISS vectorstore already exists. Loading from disk...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embedding_function,
            allow_dangerous_deserialization=True  # ✅ safe for your own vectorstore
        )
        print("[INFO] FAISS vectorstore loaded successfully!")
    else:
        print("[INFO] Creating new FAISS vectorstore and embedding chunks...")
        vectorstore = FAISS.from_texts([c.page_content for c in chunks], embedding_function)
        vectorstore.save_local(VECTORSTORE_PATH)
        print("[INFO] Vectorstore saved locally. Embeddings created for all chunks.")

    print("=== Vectorstore Builder Completed ===")
    return vectorstore

# -----------------------------
# Standalone execution
# -----------------------------
if __name__ == "__main__":
    load_dotenv()  # Load AWS_REGION, BEDROCK_EMBED_MODEL from .env

    build_vectorstore(
        embedding_model_id=os.environ.get("BEDROCK_EMBED_MODEL"),
        aws_region=os.environ.get("AWS_REGION"),
        data_folder="data"
    )