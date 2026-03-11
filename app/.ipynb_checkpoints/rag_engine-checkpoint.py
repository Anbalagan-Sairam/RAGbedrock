import os
from pathlib import Path
from langchain_aws import ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

VECTORSTORE_PATH = Path("vectorstore").resolve()

class RAGEngine:
    def __init__(self, aws_region, embedding_model_id, llm_model_id, k=3):
        self.aws_region = aws_region
        self.embedding_model_id = embedding_model_id
        self.llm_model_id = llm_model_id
        self.k = k

        # -----------------------------
        # Validate vectorstore path
        # -----------------------------
        if not VECTORSTORE_PATH.exists() or not VECTORSTORE_PATH.is_dir():
            raise RuntimeError(f"Vectorstore path {VECTORSTORE_PATH} missing or not a directory.")
        if ".." in str(VECTORSTORE_PATH):
            raise RuntimeError("Vectorstore path contains unsafe elements.")
        if not os.access(VECTORSTORE_PATH, os.R_OK):
            raise RuntimeError(f"Vectorstore path {VECTORSTORE_PATH} is not readable.")

        # Load vectorstore
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

        # Initialize LLM
        self.llm = self._init_llm()

    def _load_vectorstore(self):
        embedding_function = BedrockEmbeddings(
            model_id=self.embedding_model_id,
            region_name=self.aws_region
        )

        # Safe FAISS load (no dangerous deserialization)
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    def _init_llm(self):
        llm = ChatBedrock(
            model_id=self.llm_model_id,
            region_name=self.aws_region
        )
        return llm

    def query_llm(self, query: str):
        # Retrieve top k relevant chunks
        relevant_docs = self.retriever.get_relevant_documents(query)
        if not relevant_docs:
            return {
                "answer": "No relevant information found.",
                "retrieved_chunks": []
            }

        # Combine context
        context = "\n".join([doc.page_content for doc in relevant_docs])

        system_prompt = (
            "You are a smart assistant that answers user queries using the provided context. "
            "Always provide clear, concise, and informative answers. "
            "Answer based on context only; do not hallucinate."
        )

        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ])

        return {
            "answer": response.content,
            "retrieved_chunks": [doc.page_content for doc in relevant_docs]
        }