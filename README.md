# txt_RAG

RAG Project – Contextual AI using ETL + Vector Search:
This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch using Python, FAISS, and an LLM.
The system allows users to ask questions and receive answers grounded strictly in provided text documents.
This project is built to demonstrate ETL pipelines, semantic search, and RAG architecture as required for AI Research / Contextual Data roles.

 Project Overview:
Goal:
Enable semantic question answering over text documents using embeddings + vector search + LLM.
Key Capabilities:

Text file ingestion
Text cleaning & preprocessing
Intelligent chunking with overlap
Embedding generation (Sentence Transformers)
Vector storage using FAISS
Semantic retrieval
LLM-based answer generation (Groq)


 Architecture:
Text Document → Ingestion → Cleaning → Chunking → Embeddings → FAISS Index
                                                                    ↓
User Query → Query Embedding → Semantic Search → Context → LLM → Answer
Pipeline Flow:

Ingestion: Load text from .txt files
Cleaning: Normalize whitespace and remove special characters
Chunking: Split text into 512-token chunks with 50-token overlap
Embeddings: Generate 384-dim vectors using Sentence Transformers
Vector Store: Index embeddings in FAISS for fast similarity search
Retrieval: Find top-K most relevant chunks for user query
Context Building: Combine retrieved chunks
LLM Generation: Generate answer using Groq API (Llama 3.1)
