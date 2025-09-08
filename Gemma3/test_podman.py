#!/usr/bin/env python3
"""
Test script to verify Podman + LLaMA Server integration
Optimized for Jetson Orin Nano
"""

import json
import time
from config import PODMAN_CONFIG
from podman_manager import PodmanManager
from llama_client import LlamaClient

def test_podman_manager():
    """Test Podman container management"""
    print("Testing Podman Manager")
    print("-" * 30)
    
    manager = PodmanManager(
        container_name=PODMAN_CONFIG["container_name"],
        port=PODMAN_CONFIG["port"]
    )
    
    # Check current status
    print(f"Container exists: {manager.is_container_exists()}")
    print(f"Container running: {manager.is_container_running()}")
    
    # Get container status
    status = manager.get_container_status()
    print(f"Container status: {json.dumps(status, indent=2)}")
    
    return manager

def test_llama_client(api_url):
    """Test LLaMA Server API client"""
    print("Testing LLaMA Client")
    print("-" * 30)
    
    client = LlamaClient(api_url)
    
    # Test connection
    print(f"Server responding: {client.test_connection()}")
    
    # Get server info
    server_info = client.get_server_info()
    print(f"Server info: {json.dumps(server_info, indent=2)}")
    
    return client

def test_model_generation(client):
    """Test model generation capabilities"""
    print("Testing Model Generation")
    print("-" * 30)
    
    # Test simple chat completion
    print("Testing chat completion...")
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you say 'Hello from LLaMA Server'?"}
    ]
    
    response = client.chat_completion(test_messages)
    print(f"Chat response: {response}")
    
    # Test completion endpoint
    print("\nTesting completion...")
    completion_response = client.completion("The Jetson Nano is")
    print(f"Completion response: {completion_response}")
    
    # Test RAG-style query
    print("\nTesting RAG-style query...")
    context = "The Jetson Nano is a compact, powerful computer designed by NVIDIA for AI applications at the edge."
    query = "What is the Jetson Nano used for?"
    
    rag_response = client.ask_with_context(query, context)
    print(f"RAG response: {rag_response}")
    
    return True

def test_full_integration():
    """Test complete integration workflow"""
    print("Testing Full Integration")
    print("=" * 50)
    
    # Test Podman Manager
    manager = test_podman_manager()
    print()
    
    # Start container if not running
    if not manager.is_container_running():
        print("Starting container for testing...")
        if manager.start_container():
            print("✓ Container started successfully")
            time.sleep(5)  # Give it time to initialize
        else:
            print("✗ Failed to start container")
            print("Container logs:")
            print(manager.get_container_logs(20))
            return False
    else:
        print("✓ Container is already running")
    
    print()
    
    # Test LLaMA Client
    client = test_llama_client(PODMAN_CONFIG["api_url"])
    print()
    
    # Wait for server to be ready
    if not client.test_connection():
        print("Waiting for server to be ready...")
        for i in range(30):  # Wait up to 60 seconds
            time.sleep(2)
            if client.test_connection():
                print("✓ Server is now ready")
                break
            print(".", end="", flush=True)
        else:
            print("\n✗ Server failed to become ready")
            print("Container logs:")
            print(manager.get_container_logs(30))
            return False
    
    # Test model generation
    try:
        test_model_generation(client)
        print("\n✓ All tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Model generation test failed: {e}")
        print("Container logs:")
        print(manager.get_container_logs(20))
        return False

def main():
    print("Podman + LLaMA Server Integration Test")
    print("Optimized for Jetson Orin Nano")
    print("=" * 50)
    
    # Show configuration
    print("Configuration:")
    print(f"  Container name: {PODMAN_CONFIG['container_name']}")
    print(f"  Port: {PODMAN_CONFIG['port']}")
    print(f"  API URL: {PODMAN_CONFIG['api_url']}")
    print(f"  Model alias: {PODMAN_CONFIG['model_alias']}")
    print()
    
    # Run full integration test
    success = test_full_integration()
    
    if success:
        print("\n" + "=" * 50)
        print("All tests passed! Your Podman setup is ready.")
        print("You can now run:")
        print("  python assistant_podman.py    # Full voice assistant")
        print("  python demo_text_podman.py    # Text-only demo")
    else:
        print("\n" + "=" * 50)
        print("Some tests failed. Please check the errors above.")
        print("Make sure:")
        print("  1. Podman is installed and working")
        print("  2. Model files exist at the specified paths")
        print("  3. Container image is available")
        print("  4. No other services are using port", PODMAN_CONFIG['port'])

if __name__ == "__main__":
    main()