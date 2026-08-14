# Zepto Support Assistant

An offline RAG-based customer support assistant for Zepto policy questions.

## Project Structure

support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── chroma_db/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md

## RAG Architecture

Policy Documents -> Ingestion -> Embedding -> ChromaDB -> Retrieval -> Generation

### Ingestion
The eight Zepto policy documents are stored in the docs directory.

### Embedding
The project uses all-MiniLM-L6-v2 from sentence-transformers.

### Vector Store
ChromaDB stores embeddings in the zepto_policies collection.

### Retrieval
Policy questions are embedded and the top three similar documents are retrieved.

### LangGraph
The graph contains three required nodes:

1. classify_intent
2. retrieve_and_answer
3. direct_answer

Routing:

classify_intent -> policy_question -> retrieve_and_answer
classify_intent -> general_question -> direct_answer

## MOCK_LLM Mode

MOCK_LLM=1 is the default offline graded mode.

In mock mode:
- No external LLM API is called.
- Intent classification uses keyword matching.
- Policy questions use ChromaDB retrieval.
- Responses are generated deterministically from retrieved context.
- General questions return a fixed response.
- Pydantic validates the final response.

## API

### Endpoint

POST /ask

### Request

{"query": "What is the delivery fee for orders below INR 149?"}

### Policy Response

{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_03"
  ],
  "confidence": 1.0
}

### General Question

{"query": "What is the capital of France?"}

Response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

## Pydantic Response Schema

{"answer": "string", "sources": ["document_id"], "confidence": 1.0}

Confidence is restricted to the range 0 to 1.

## Running Locally

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn main:app --host 0.0.0.0 --port 7860

API endpoint:

http://127.0.0.1:7860/ask

## Docker

Build:

docker build -t zepto-support-assistant .

Run:

docker run -p 7860:7860 zepto-support-assistant

## Key Components

Ingestion: docs/
Embedding: all-MiniLM-L6-v2
Vector Store: ChromaDB
Intent Routing: classify_intent
Retrieval: retrieve_and_answer
Direct Response: direct_answer
Validation: Pydantic
API: FastAPI /ask
Containerization: Dockerfile

## Offline Design

The graded baseline runs fully offline with MOCK_LLM=1.
Embeddings are generated locally and vectors are stored in local ChromaDB.
No external LLM API is required.