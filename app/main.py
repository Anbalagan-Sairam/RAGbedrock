# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import json
import boto3

from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

app = FastAPI(title="Bedrock RAG Demo", version="0.1.0")

# -------------------------------
# Request/Response models
# -------------------------------
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []

# -------------------------------
# AWS Bedrock runtime client
# -------------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# -------------------------------
# Embeddings wrapper
# -------------------------------
class BedrockEmbeddings(Embeddings):
    model_id = "amazon.titan-embed-text-v1"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            payload = {"inputText": text}  # correct Titan schema
            resp = bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )
            vectors.append(json.loads(resp["body"].read())["embedding"])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embeddings = BedrockEmbeddings()

# -------------------------------
# Load documents
# -------------------------------
documents = []
docs_folder = "docs"
for fname in os.listdir(docs_folder):
    if fname.endswith(".txt"):
        with open(os.path.join(docs_folder, fname), "r") as f:
            text = f.read()
        documents.append(Document(page_content=text, metadata={"source": fname}))

if not documents:
    raise ValueError("No .txt files found in docs folder.")

# -------------------------------
# Build FAISS vectorstore (let LangChain handle index)
# -------------------------------
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

# -------------------------------
# Nova Lite LLM wrapper
# -------------------------------
class NovaLiteLLM(LLM):
    @property
    def _llm_type(self):
        return "novalite"

    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "schemaVersion": "messages-v1",
            "messages": [{"role": "user", "content": [{"text": prompt}]}]
        }
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        content_list = result.get("output", {}).get("message", {}).get("content", [])
        if content_list:
            return content_list[0].get("text", "")
        return "No answer returned from model"

llm = NovaLiteLLM()

# -------------------------------
# RetrievalQA chain
# -------------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# /ask endpoint
# -------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = qa_chain({"query": request.query})
        answer = result["result"]
        sources = [doc.metadata["source"] for doc in result["source_documents"]]
    except Exception as e:
        answer = f"Error calling RAG system: {e}"
        sources = []

    return AskResponse(answer=answer, sources=sources)