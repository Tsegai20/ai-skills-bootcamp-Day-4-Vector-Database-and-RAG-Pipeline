
# Day 4 — Vector Database and RAG Pipeline

## What I built
A full Retrieval Augmented Generation system over a personal
research knowledge base using ChromaDB and Groq LLaMA.

## How it works
1. Documents embedded using sentence-transformers all-MiniLM-L6-v2
2. Embeddings stored in ChromaDB with cosine similarity
3. User question converted to embedding vector
4. ChromaDB retrieves top N most similar documents
5. Retrieved documents fed as context to LLM
6. LLM generates grounded answer citing only retrieved context

## Key Features
- Semantic search finds meaning not just keywords
- Grounded answers — LLM says I cannot answer if context is insufficient
- Adjustable number of retrieved documents
- Source document display with similarity scores
- Knowledge base browser
- Example questions sidebar

## Tech Stack
- ChromaDB — vector database
- Sentence Transformers — text embeddings
- Groq LLaMA — answer generation
- Streamlit — web interface

## What I learned
- Vector embeddings and semantic similarity
- ChromaDB collection management
- RAG pipeline architecture
- Cosine similarity vs L2 distance
- Grounding LLMs to prevent hallucination
- Context window management

## Author
Tsegai Yhdego
PhD Industrial Engineering — FAMU-FSU
AI/ML Researcher — R-SEAT Center
