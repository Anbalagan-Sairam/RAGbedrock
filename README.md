10/03/2025:
v0.1 Adding skeletal working PoC to get the RAG system working

We use uvicorn web server that listens to HTTP requests, passes them to FASTAPI app and sends the response back to client.

uvicorn app.main:app --host 0.0.0.0 --port 8000