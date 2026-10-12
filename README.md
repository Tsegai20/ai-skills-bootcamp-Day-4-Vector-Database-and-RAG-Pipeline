# Day 4 — Vector Database and RAG Pipeline

**AI Skills Bootcamp | Tsegai Yhdego, PhD**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA-orange)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)](https://streamlit.io)

---

## What I Built

A full Retrieval Augmented Generation system over a personal research knowledge base. The system answers questions grounded in real documents — and correctly says "I cannot answer" when the context is insufficient, preventing hallucination.

---

## How RAG Works
User Question
↓
Convert to 384-dimensional embedding vector
↓
ChromaDB cosine similarity search
↓
Retrieve top N most relevant document chunks
↓
Build prompt: question + retrieved context
↓
Groq LLaMA generates grounded answer
↓
Answer with source citations


---

## Knowledge Base

10 research documents covering:

| Document | Topic |
|----------|-------|
| Federated Learning for Manufacturing | Privacy-preserving multi-site ML |
| Differential Privacy in ML | Mathematical privacy framework |
| UAV Route Optimization | Autonomous navigation algorithms |
| Airline Delay Propagation | LSTM/RNN forecasting |
| Ontology-Based Attribute Learning | Zero-shot defect detection |
| Intelligent Transportation Systems | USDOT research overview |
| Vehicle Sales Forecasting | GM Financial ML pipelines |
| Transfer Learning and BERT | NLP fine-tuning concepts |
| Computer Vision for Manufacturing | CNN defect detection |
| MLOps and Production Pipelines | Kubeflow, MLflow, MinIO |

---

## Key Features

- **Semantic search** — finds meaning not just keywords. "drone path planning" finds UAV navigation documents
- **Grounded answers** — LLM stays anchored to retrieved context, not hallucinated knowledge
- **Hallucination prevention** — system correctly says "I cannot answer" when context is insufficient
- **Adjustable retrieval** — slider controls how many documents feed into the answer
- **Source display** — shows which documents were used with similarity scores
- **Knowledge base browser** — read any document directly in the app
- **Example questions** — sidebar with pre-built questions about the research

---

## The Critical Test

Question: “What is the best recipe for chocolate cake?”

Retrieved: Federated Learning doc (8%), UAV doc (6%), Ontology doc (2%)

Answer: “The provided documents do not contain any information
about chocolate cake recipes. Therefore, I cannot answer
that question based on the given context.”


**This is the correct behavior.** A system that says "I don't know" is far safer than one that hallucinates an answer — especially in healthcare, legal, or financial applications.

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Text embeddings | sentence-transformers all-MiniLM-L6-v2 → 384-dim vectors |
| Cosine similarity | Measures angle between vectors — closer = more similar |
| ChromaDB | Stores and searches embeddings with HNSW index |
| Semantic search | Query embedding matched against document embeddings |
| RAG pipeline | Retrieve → Augment prompt → Generate grounded answer |
| Hallucination prevention | LLM instructed to answer only from context |
| Context window | Top N chunks fed to LLM within token limit |

---

## Semantic Search vs Keyword Search

| Query | Keyword Search | Semantic Search |
|-------|---------------|-----------------|
| "drone path planning" | No match (word "drone" not in docs) | Finds UAV navigation doc ✅ |
| "privacy machine learning" | Finds only exact phrase | Finds federated learning + differential privacy ✅ |
| "delay airline network" | Misses synonym matches | Finds propagation forecasting doc ✅ |

---

## Extension — PDF and URL Ingestion

Built a second app (`app_rag_docs.py`) that ingests any PDF or URL:

Upload PDF or paste URL
↓
Extract text (PyMuPDF for PDFs, BeautifulSoup for URLs)
↓
Chunk text (500 chars, 50 char overlap)
↓
Embed each chunk with sentence-transformers
↓
Store in ChromaDB
↓
Ask questions about your documents


---

## Project Structure

day4/
├── app_day4.py # RAG app over research knowledge base
├── app_rag_docs.py # PDF and URL document ingestion app
├── requirements_day4.txt
└── README.md


---

## Run Locally

```bash
git clone https://github.com/Tsegai20/ai-skills-bootcamp-day4.git
cd ai-skills-bootcamp-day4
pip install -r requirements_day4.txt

# Research knowledge base RAG
streamlit run app_day4.py

# PDF and URL document RAG
streamlit run app_rag_docs.py
```

**API:** Requires free Groq API key from console.groq.com

---

## Skills Added to Resume

- RAG pipeline development
- ChromaDB vector database
- Sentence Transformers embeddings
- Semantic search implementation
- LLM grounding and hallucination prevention
- PDF and URL document ingestion
- Text chunking strategies
- Context window management

---

## Part of AI Skills Bootcamp

| Day | Project | Repo |
|-----|---------|------|
| 1 | Streamlit AI Apps | [Day 1](https://github.com/Tsegai20/ai-skills-bootcamp) |
| 2 | Fine-Tuned BERT Classifier | [Day 2](https://github.com/Tsegai20/ai-skills-bootcamp) |
| 3 | FastAPI ML Serving | [Day 3](https://github.com/Tsegai20/ai-skills-bootcamp) |
| **4** | **RAG Pipeline** | **You are here** |
| 5 | MLflow + Federated Learning | Coming |
| 6 | Docker Federated System | [ai-skills-bootcamp6](https://github.com/Tsegai20/ai-skills-bootcamp6) |

---

## Author

**Tsegai Yhdego, PhD**  
Industrial Engineering — Machine Learning and Optimization  
Assistant Director, R-SEAT Center, FAMU-FSU  
USDOT-Funded Research — Intelligent Transportation and Autonomous Systems  
[GitHub](https://github.com/Tsegai20) · [LinkedIn](https://www.linkedin.com/in/tsegai-yhdego-ph-d-b6209b118/)
