# app/api.py
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.config import AppConfig
from app.rag_engine import RAGEngine
import boto3
from botocore.exceptions import ClientError

# -----------------------------
# Load and validate config
# -----------------------------
try:
    config = AppConfig()
except Exception as e:
    print(f"[FATAL] Configuration validation failed: {e}")
    sys.exit(1)

# -----------------------------
# Bedrock connectivity validation
# -----------------------------
try:
    client = boto3.client("bedrock", region_name=config.aws_region)
    # Minimal call to check connectivity
    client.list_foundation_models()  
except ClientError as e:
    print(f"[FATAL] Unable to access Bedrock in region '{config.aws_region}': {e}")
    sys.exit(1)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(
    title="TrailblazeAI RAG API",
    description="RAG system with FastAPI endpoints and startup validation",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# Request schema
# -----------------------------
class QueryRequest(BaseModel):
    query: str

# -----------------------------
# Global RAG engine (initialized once)
# -----------------------------
rag_engine: RAGEngine = None

@app.on_event("startup")
def startup_event():
    global rag_engine
    print("[INFO] Initializing RAG Engine on startup...")
    try:
        rag_engine = RAGEngine(
            aws_region=config.aws_region,
            embedding_model_id=config.bedrock_embed_model,
            llm_model_id=config.bedrock_llm_model,
            k=config.model_k
        )
    except Exception as e:
        print(f"[FATAL] Failed to initialize RAG Engine: {e}")
        sys.exit(1)
    print("[INFO] RAG Engine ready!")

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -----------------------------
# Ready check (RAG loaded)
# -----------------------------
@app.get("/ready")
def ready():
    if rag_engine is None:
        return {"ready": False}
    return {"ready": True}

# -----------------------------
# Query endpoint
# -----------------------------
@app.post("/query")
def query(request: QueryRequest):
    if rag_engine is None:
        return {"error": "RAG engine not ready"}
    
    result = rag_engine.query_llm(request.query)
    return {
        "question": request.query,
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"]
    }