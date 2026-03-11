# app/main.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock, BedrockEmbeddings

VECTORSTORE_PATH = "vectorstore"

# -----------------------------
# Load AWS env variables
# -----------------------------
load_dotenv()  # AWS_REGION, BEDROCK_LLM_MODEL, BEDROCK_EMBED_MODEL

aws_region = os.environ.get("AWS_REGION")
embedding_model_id = os.environ.get("BEDROCK_EMBED_MODEL")
llm_model_id = os.environ.get("BEDROCK_LLM_MODEL")

# -----------------------------
# 1️⃣ Load FAISS vectorstore
# -----------------------------
print("[INFO] Loading FAISS vectorstore...")
embedding_function = BedrockEmbeddings(
    model_id=embedding_model_id,
    region_name=aws_region
)
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embedding_function,
    allow_dangerous_deserialization=True
)
print("[INFO] FAISS vectorstore loaded successfully!")

# -----------------------------
# 2️⃣ Create retriever
# -----------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # top 3 relevant chunks
)

# -----------------------------
# 3️⃣ Initialize BedrockChat LLM
# -----------------------------
llm = ChatBedrock(
    model_id=llm_model_id,
    region_name=aws_region
)

# -----------------------------
# 4️⃣ Query loop with chunk logging
# -----------------------------
print("\nTrailblazeAI RAG Ready! Type 'exit' to quit.")
while True:
    query = input("\nQuestion: ")
    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        print("No relevant information found for this query.")
        continue

    # -----------------------------
    # Log retrieved chunks
    # -----------------------------
    print(f"[INFO] Retrieved {len(relevant_docs)} chunks for query '{query}':")
    for i, doc in enumerate(relevant_docs, 1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  Chunk {i} preview: {preview} ...")

    # Combine context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Ask LLM
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer this question based on the following context:\n{context}\n\nQuestion:\n{query}"}
    ])

    print("\nAnswer:\n", response.content)