#!/usr/bin/env python3
"""
Test script for PDF RAG functionality
Optimized for Jetson Orin Nano
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from assistant_ollama import (
    VectorDatabase, PDFProcessor, 
    add_pdf_to_knowledge_base, load_pdfs_from_directory,
    get_rag_stats, clear_rag_cache, rag_ask
)

def create_sample_pdf_directory():
    """Create sample PDF directory for testing"""
    pdf_dir = Path(PDF_CONFIG["pdf_directory"])
    pdf_dir.mkdir(exist_ok=True)
    
    # Create a simple text file as placeholder
    sample_file = pdf_dir / "README.txt"
    if not sample_file.exists():
        with open(sample_file, 'w') as f:
            f.write("""
PDF RAG Test Directory

Place your PDF files in this directory for automatic processing.
The voice assistant will automatically load and index these PDFs
for enhanced question answering capabilities.

Supported formats: .pdf
Maximum file size: 50MB
            """)
    
    print(f"Created PDF directory: {pdf_dir}")
    return pdf_dir

def test_pdf_processor():
    """Test PDF processor functionality"""
    print("\n=== Testing PDF Processor ===")
    
    processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
    
    # Test text cleaning
    dirty_text = "This  is   a    test\n\n\nwith    extra   spaces!!!"
    clean_text = processor.clean_text(dirty_text)
    print(f"Text cleaning: '{dirty_text}' -> '{clean_text}'")
    
    # Test chunking
    test_text = " ".join(["word"] * 200)  # 200 words
    chunks = processor.chunk_text(test_text)
    print(f"Chunking: 200 words -> {len(chunks)} chunks")
    
    return True

def test_vector_database():
    """Test vector database functionality"""
    print("\n=== Testing Vector Database ===")
    
    # Create test database
    test_db = VectorDatabase(dim=384, cache_dir="./test_rag_cache")
    
    # Add test documents
    test_docs = [
        "The Jetson Orin Nano is a powerful edge AI computer.",
        "Ollama provides local language model serving capabilities.",
        "FAISS enables efficient similarity search for embeddings."
    ]
    
    test_db.add_documents(test_docs, source="test")
    
    # Test search
    results = test_db.search("Jetson AI computer", top_k=2)
    print(f"Search results: {len(results)} found")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.3f}, Source: {result['source']}")
        print(f"      Text: {result['text'][:60]}...")
    
    # Test stats
    stats = test_db.get_stats()
    print(f"Database stats: {stats}")
    
    # Cleanup
    import shutil
    if os.path.exists("./test_rag_cache"):
        shutil.rmtree("./test_rag_cache")
    
    return True

def test_ollama_connection():
    """Test Ollama connection before RAG testing"""
    print("\n=== Testing Ollama Connection ===")
    
    import requests
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"Ollama is running with {len(models)} models")
            
            # Check if our model is available
            model_names = [m['name'] for m in models]
            if MODEL_NAME in model_names:
                print(f"Model {MODEL_NAME} is available")
                return True
            else:
                print(f"Model {MODEL_NAME} not found. Available: {model_names}")
                print(f"Download with: ollama pull {MODEL_NAME}")
                return False
        else:
            print(f"Ollama error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama. Start with: ollama serve")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_rag_functionality():
    """Test complete RAG functionality"""
    print("\n=== Testing RAG Functionality ===")
    
    # Get current stats
    stats = get_rag_stats()
    print(f"Current knowledge base: {stats}")
    
    # Test queries
    test_queries = [
        "What is the Jetson Nano?",
        "How does Ollama work?",
        "What is RAG?",
        "Tell me about AI development"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            # Test with timeout since Ollama might be slow
            response = rag_ask(query)
            print(f"Response: {response[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    return True

def main():
    """Main test function"""
    print("PDF RAG Test Suite for Jetson Orin Nano")
    print("=" * 50)
    
    # Create necessary directories
    pdf_dir = create_sample_pdf_directory()
    
    # Run tests
    tests = [
        ("PDF Processor", test_pdf_processor),
        ("Vector Database", test_vector_database),
        ("Ollama Connection", test_ollama_connection),
        ("RAG Functionality", test_rag_functionality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results[test_name] = "ERROR"
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗" if result == "FAIL" else "⚠"
        print(f"{status_symbol} {test_name}: {result}")
    
    # Usage instructions
    print("\n" + "=" * 50)
    print("USAGE INSTRUCTIONS")
    print("=" * 50)
    print(f"1. Place PDF files in: {pdf_dir}")
    print("2. Start Ollama: ollama serve")
    print(f"3. Download model: ollama pull {MODEL_NAME}")
    print("4. Run assistant: python assistant_ollama.py")
    print("5. Ask questions about your PDFs!")
    
    print("\nPDF RAG Commands:")
    print("  from assistant_ollama import *")
    print("  add_pdf_to_knowledge_base('path/to/file.pdf')")
    print("  load_pdfs_from_directory('./pdfs')")
    print("  get_rag_stats()")
    print("  clear_rag_cache()  # Reset database")

if __name__ == "__main__":
    main()