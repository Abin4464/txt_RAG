# main.py - TXT Version

import os
import sys

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Add to Python path
sys.path.insert(0, SRC_DIR)

print(f"Looking for modules in: {SRC_DIR}")
print(f"Does src exist? {os.path.exists(SRC_DIR)}")
print(f"Does txt_loader.py exist? {os.path.exists(os.path.join(SRC_DIR, 'ingestion', 'txt_loader.py'))}")

# Now import - CHANGED: Using TXTLoader instead of PDFLoader
try:
    from ingestion.txt_loader import TXTLoader  # ← CHANGED
    from preprocessing.cleaner import TextCleaner
    from preprocessing.chunker import TextChunker
    from embeddings.embedder import Embedder
    from vectorstore.faiss_store import FAISSStore
    from retrieval.search import Retriever
    from llm.llm_client import LLMClient
    from utils.config import Config
    
    print("\n✅ All imports successful!\n")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}\n")
    sys.exit(1)

def main():
    print("=" * 60)
    print("RAG PIPELINE - TXT VERSION")  # ← CHANGED
    print("Using: Sentence Transformers + Groq + FAISS")
    print("=" * 60)
    
    # Step 1: Ingestion - CHANGED for TXT
    print("\n[1/8] INGESTION - Loading TXT...")
    txt_path = os.path.join(BASE_DIR, "data", "documents", "sample.txt")  # ← CHANGED
    
    if not os.path.exists(txt_path):
        print(f"❌ TXT not found at: {txt_path}")
        print("Please add a TXT file to data/documents/sample.txt")  # ← CHANGED
        return
    
    loader = TXTLoader(txt_path)  # ← CHANGED
    raw_text = loader.load()
    
    if not raw_text:
        print("Failed to load TXT. Exiting.")
        return
    
    # Step 2: Cleaning
    print("\n[2/8] CLEANING - Normalizing text...")
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_text)
    
    # Step 3: Chunking
    print("\n[3/8] CHUNKING - Splitting into chunks...")
    chunker = TextChunker(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunks = chunker.chunk(cleaned_text)
    
    # Step 4: Embeddings (FREE - Sentence Transformers)
    print("\n[4/8] EMBEDDINGS - Generating with Sentence Transformers...")
    embedder = Embedder(model_name=Config.EMBEDDING_MODEL)
    embeddings = embedder.embed(chunks)
    
    # Step 5: Vector Database
    print("\n[5/8] VECTOR DATABASE - Indexing embeddings...")
    vectorstore = FAISSStore(dimension=384)
    vectorstore.add(embeddings, chunks)
    
    # Step 6: Retrieval
    print("\n[6/8] RETRIEVAL - Setting up retriever...")
    retriever = Retriever(
        vectorstore=vectorstore,
        embedder=embedder,
        top_k=Config.TOP_K_RESULTS
    )
    
    # Step 7: LLM (FREE - Groq)
    print("\n[7/8] LLM - Setting up Groq client...")
    llm = LLMClient(
        api_key=Config.GROQ_API_KEY,
        model=Config.LLM_MODEL
    )
    
    # Step 8: Query Loop
    print("\n[8/8] READY - You can now query the system")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        query = input("\n💬 Enter your question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        # Retrieve relevant chunks
        results = retriever.retrieve(query)
        
        # Build context
        context = retriever.build_context(results)
        
        # Generate response
        print("\n📝 Answer:")
        answer = llm.generate_response(query, context)
        print(answer)
        
        # Show retrieved chunks
        print("\n📚 Retrieved Chunks:")
        for i, (text, distance) in enumerate(results, 1):
            print(f"\n  [{i}] Distance: {distance:.4f}")
            print(f"  {text[:200]}...")

if __name__ == "__main__":
    main()