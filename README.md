# Zepto-Data-AI-Platform

An end-to-end quick-commerce engineering ecosystem integrating distributed data pipelines, predictive analytics, and an offline-first GenAI retrieval-augmented generation (RAG) service.

---

## Repository Overview

This repository houses three core modules powering quick commerce operations:

1. **Module 1: Data Pipeline (`/data_pipeline`)**  
   Scalable data processing and ingestion pipelines simulating high-throughput order, inventory, and fulfillment streams.

2. **Module 2: Analytics & Predictive Modeling (`/analytics`)**  
   Exploratory data analysis (EDA), customer segmentation, and predictive machine learning models evaluating delivery times, driver allocation, and order behaviors.

3. **Module 3: Support Assistant (`/support_assistant`)**  
   An offline-first, LangGraph-orchestrated customer support RAG pipeline backed by local vector embeddings, structured Pydantic schema validation, and a containerized FastAPI service.

---

## 📂 Repository Architecture

```text
Zepto-Data-AI-Platform/
├── analytics/                      # Module 2: Analytics & Machine Learning
│   ├── notebooks/                  # EDA & modeling experiments
│   └── README.md                   # Analytics methodology & metrics
├── data_pipeline/                  # Module 1: Data Pipeline Services
│   └── README.md                   # ETL/ELT pipelines & orchestration
├── support_assistant/              # Module 3: GenAI RAG Assistant
│   ├── chroma_db/                  # Persistent ChromaDB vector store
│   ├── docs/                       # 8 Official Zepto policy corpus documents
│   ├── __init__.py
│   ├── Dockerfile                  # Production container configuration
│   ├── graph.py                    # LangGraph StateGraph & conditional routing
│   ├── indexer.py                  # Local chunking & all-MiniLM-L6-v2 embedding
│   ├── main.py                     # FastAPI service exposing POST /ask
│   ├── prompts.py                  # Structured Prompt Templates
│   └── README.md                   # Detailed RAG architecture & verification
├── .gitignore
├── README.md                       # Platform root documentation
└── requirements.txt                # Unified dependency definitions





## Module 3 Deep Dive: Support Assistant RAG Pipeline
Module 3 delivers an offline-first, deterministic RAG system for customer policy resolution without requiring external API keys or paid third-party dependencies.

Architecture Flow
                       +----------------------+
                       |    Customer Query    |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       |   classify_intent    |
                       | (Keyword Router Node)|
                       +----------+-----------+
                                  |
                     +------------+------------+
                     |                         |
        [policy_question]              [general_question]
                     |                         |
                     v                         v
        +------------------------+   +--------------------+
        |  retrieve_and_answer   |   |   direct_answer    |
        | (all-MiniLM-L6-v2 +    |   |  (Canned String)   |
        |   ChromaDB Top-3)      |   +---------+----------+
        +------------+-----------+             |
                     |                         |
                     +------------+------------+
                                  |
                                  v
                       +----------------------+
                       | Validated Response   |
                       |  (Pydantic Schema)   |
                       +----------------------+


Core Components
Ingestion & Embedding (indexer.py): Indexes 8 delivery, returns, membership, and support policy documents (doc_01.txt through doc_08.txt) locally using sentence-transformers/all-MiniLM-L6-v2.

Vector Store: Persisted in ChromaDB (zepto_policy_collection) using cosine similarity.

LangGraph State Machine (graph.py):

classify_intent: Heuristic intent router distinguishing policy queries from general queries.

retrieve_and_answer: Fetches top-3 relevant context chunks for policy inquiries.

direct_answer: Handles out-of-scope queries gracefully with a deterministic canned response.

API Wrapper (main.py): Exposes a validated POST /ask endpoint returning structured JSON (answer, sources, confidence).

Containerization (Dockerfile): Containerized FastAPI service ready for local deployment on port 7860.




## Quickstart Guide
1. Clone & Set Up Environment

git clone [https://github.com/prasad-b-code/Zepto-Data-AI-Platform.git](https://github.com/prasad-b-code/Zepto-Data-AI-Platform.git)
cd Zepto-Data-AI-Platform

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt



2. Build ChromaDB Vector Index
  python support_assistant/indexer.py


3. Run Support Assistant API
 python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860 --reload

Interactive API Docs (Swagger): Visit http://localhost:7860/docs

Health Check: http://localhost:7860/health


4. Run via Docker
 # Build Docker image
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .

# Run container
docker run -p 7860:7860 zepto-support-assistant



Example API Calls (MOCK_LLM=1)
Policy Query (Triggering Vector Retrieval)
BASH
 curl -X POST "[http://127.0.0.1:7860/ask](http://127.0.0.1:7860/ask)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee for orders below 149?"}'

JSON
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard deliv...",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_05"
  ],
  "confidence": 1.0
}


General Query (Direct Response Route)
BASH
 curl -X POST "[http://127.0.0.1:7860/ask](http://127.0.0.1:7860/ask)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'


JSON
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}



