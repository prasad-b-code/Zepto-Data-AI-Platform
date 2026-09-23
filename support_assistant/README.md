# Zepto Support Assistant (`/support_assistant`)

An offline-first, LangGraph-orchestrated Retrieval-Augmented Generation (RAG) assistant for Zepto's delivery, return, and membership policies.

---

##1. RAG Pipeline Architecture

```text
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
               | (Pydantic Schema)    |
               +----------------------+
```

### Stage-by-Stage Breakdown

1. **Ingestion (`support_assistant/indexer.py: build_index`)**:
   Loads the 8 Zepto policy text documents (`doc_01.txt` through `doc_08.txt`) from `/support_assistant/docs/`. Each document is treated as an atomic chunk with its document ID and source filename preserved in metadata.

2. **Embedding (`all-MiniLM-L6-v2`)**:
   Locally embeds each chunk into a 384-dimensional dense vector space using `sentence-transformers` with the `all-MiniLM-L6-v2` model. This runs entirely on local CPU with zero API keys or external network requests.

3. **Retrieval (`ChromaDB`)**:
   Vectors and text chunks are persisted in a local ChromaDB collection named `zepto_policy_collection` under `support_assistant/chroma_db/`. Incoming queries are converted to vector embeddings and matched against document vectors using cosine similarity to return the top 3 chunks.

4. **Intent Routing & Generation (`LangGraph StateGraph` in `support_assistant/graph.py`)**:
   - `classify_intent`: Evaluates whether the query requires policy context using keyword routing (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`).
   - `retrieve_and_answer`: Triggers ChromaDB retrieval and formats a grounded response based on the top retrieved chunk.
   - `direct_answer`: Handles general questions by returning a standard policy-only disclaimer without querying the vector store.

5. **Structured Schema Validation (`Pydantic: QueryResponse`)**:
   Enforces a deterministic output contract with fields: `answer` (string), `sources` (list of document IDs), and `confidence` (float between 0.0 and 1.0).

---

## 2. MOCK_LLM Toggle Behavior

| Pipeline Stage | Default Graded Mock Baseline (`MOCK_LLM=1` or unset) | Optional Real LLM Mode (`MOCK_LLM=0`) |
| :--- | :--- | :--- |
| **Document Ingestion & Indexing** | Runs locally with `all-MiniLM-L6-v2` into persistent ChromaDB. | Identical local execution; no external API needed. |
| **Vector Retrieval** | Computes cosine similarity in ChromaDB returning top-3 nearest chunks. | Identical local retrieval. |
| **Intent Classification** | Deterministic keyword check across 8 policy keywords. | Prompts LLM to classify query intent. |
| **Policy Answer Generation** | Deterministic template: `f"Based on the retrieved context: {top_chunk_snippet}"`. | Prompts LLM using structured role-context-task template with retry validation. |
| **General Query Generation** | Fixed string: `"I can only answer questions about Zepto policies right now."`. | Prompts LLM directly without retrieval context. |

---

## 3. Verified Example API Calls (`MOCK_LLM=1`)

### Call 1: Policy Question (Retrieval Triggered)

**Endpoint:** `POST /ask`  
**Request Payload:**
```json
{
  "query": "What is the delivery fee for orders below 149?"
}
```

**Response Payload:**
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard deliv...",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_05"
  ],
  "confidence": 1.0
}
```

### Call 2: General Question (Direct Answer, No Retrieval)

**Endpoint:** `POST /ask`  
**Request Payload:**
```json
{
  "query": "What is the capital of France?"
}
```

**Response Payload:**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## 4. Local Run & Container Commands

### Local Execution (Uvicorn)
```powershell
python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker Build & Run
```powershell
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```