#!/usr/bin/env python3
"""
Text-based demo of the Podman + LLaMA Server assistant without audio requirements
Optimized for Jetson Orin Nano
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import *
from podman_manager import PodmanManager
from llama_client import LlamaClient

# Load sentence transformer model for document embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Documents to be used in Retrieval-Augmented Generation (RAG)
docs = KNOWLEDGE_DOCS

# Vector Database class (simplified version)
class VectorDatabase:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
    
    def add_documents(self, docs):
        embeddings = embedding_model.encode(docs)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(docs)
    
    def search(self, query, top_k=3):
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.documents[i] for i in indices[0]]

# Create a VectorDatabase and add documents to it
db = VectorDatabase(dim=384)
db.add_documents(docs)

def rag_ask(query, llama_client):
    """Generate a response using Retrieval-Augmented Generation (RAG)"""
    context = " ".join(db.search(query))
    return llama_client.ask_with_context(query, context)

def main():
    print("Podman + LLaMA Server Assistant Demo")
    print("Optimized for Jetson Orin Nano")
    print("=" * 50)
    
    # Initialize Podman manager
    print("Initializing Podman deployment...")
    podman_manager = PodmanManager(
        container_name=PODMAN_CONFIG["container_name"],
        port=PODMAN_CONFIG["port"]
    )
    
    # Check if container is running
    if not podman_manager.is_container_running():
        print("Container not running. Starting...")
        if podman_manager.start_container():
            print("✓ Container started successfully")
        else:
            print("✗ Failed to start container")
            print("Recent logs:")
            print(podman_manager.get_container_logs(10))
            return
    else:
        print("✓ Container is already running")
    
    # Initialize LLaMA client
    llama_client = LlamaClient(PODMAN_CONFIG["api_url"])
    
    # Test connection
    if not llama_client.test_connection():
        print("✗ Failed to connect to LLaMA server")
        print("Container logs:")
        print(podman_manager.get_container_logs(20))
        return
    else:
        print("✓ Connected to LLaMA server")
    
    # Show server info
    server_info = llama_client.get_server_info()
    print(f"Server status: {server_info.get('status', 'unknown')}")
    print(f"Available models: {[m.get('id') for m in server_info.get('models', [])]}")
    
    print("-" * 50)
    print("This is a text-based demo. Type your questions and press Enter.")
    print("Type 'quit' to exit.")
    print("Type 'status' to check deployment status.")
    print("Type 'logs' to see container logs.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🎤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'status':
                status = podman_manager.get_container_status()
                print(f"Container status: {status}")
                server_info = llama_client.get_server_info()
                print(f"Server info: {server_info}")
                continue
            
            if user_input.lower() == 'logs':
                logs = podman_manager.get_container_logs(20)
                print("Recent container logs:")
                print(logs)
                continue
            
            if not user_input:
                print("Assistant: Please say something!")
                continue
            
            # Get context from RAG
            context_docs = db.search(user_input)
            print(f"Context found: {len(context_docs)} relevant documents")
            
            # Generate response
            print("Thinking...")
            response = rag_ask(user_input, llama_client)
            
            print(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()