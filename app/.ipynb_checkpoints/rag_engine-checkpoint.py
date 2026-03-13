# app/rag_engine.py

import os
from pinecone import Pinecone
from langchain.schema import Document
from langchain_aws import BedrockEmbeddings, ChatBedrock

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "trailblazeai")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "amazon.titan-chat")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing")

class RAGEngine:

    def __init__(self, top_k: int = 3):

        self.top_k = top_k

        # Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(INDEX_NAME)

        # Bedrock embedding model
        self.embedder = BedrockEmbeddings(
            model_id=BEDROCK_EMBED_MODEL,
            region_name=AWS_REGION
        )

        # Bedrock LLM
        self.llm = ChatBedrock(
            model_id=BEDROCK_LLM_MODEL,
            region_name=AWS_REGION
        )

    def query(self, query_text: str):

        # Embed query
        vector = self.embedder.embed_query(query_text)

        # Search Pinecone
        result = self.index.query(
            vector=vector,
            top_k=self.top_k,
            include_metadata=True
        )

        docs = []

        for match in result["matches"]:
            text = match["metadata"].get("text", "")
            if text:
                docs.append(Document(page_content=text))

        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query_text}
"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "retrieved_chunks": [d.page_content for d in docs]
        }