#!/usr/bin/env python3
"""
LLaMA Server API Client
Compatible with llama-server (llama.cpp) served via Podman container
Replaces Ollama API calls with OpenAI-compatible API calls
"""

import requests
import json
from typing import Dict, List, Optional, Any
from config import GENERATION_OPTIONS, INITIAL_PROMPT

class LlamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """
        Initialize LLaMA client for llama-server API
        
        Args:
            base_url: Base URL for the llama-server API
        """
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.completions_url = f"{self.base_url}/v1/completions"
        self.health_url = f"{self.base_url}/health"
        self.models_url = f"{self.base_url}/v1/models"
        
        # Map Ollama generation options to OpenAI format
        self.default_options = {
            "max_tokens": GENERATION_OPTIONS.get("num_predict", 80),
            "temperature": GENERATION_OPTIONS.get("temperature", 0.7),
            "top_p": GENERATION_OPTIONS.get("top_p", 0.9),
            "stream": False
        }
    
    def test_connection(self) -> bool:
        """Test if the server is responding"""
        try:
            response = requests.get(self.health_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_models(self) -> List[Dict]:
        """Get available models from the server"""
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                print(f"Error getting models: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error connecting to server: {e}")
            return []
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "ggml-org/gemma-3-4b-it-GGUF",
                       **kwargs) -> str:
        """
        Send a chat completion request (OpenAI-compatible)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (uses the alias from container)
            **kwargs: Additional parameters
        
        Returns:
            Generated response text
        """
        # Merge default options with provided kwargs
        options = {**self.default_options, **kwargs}
        
        data = {
            "model": model,
            "messages": messages,
            **options
        }
        
        try:
            response = requests.post(
                self.chat_url, 
                json=data, 
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                else:
                    return "No response generated"
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to LLaMA server. Make sure the Podman container is running."
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be taking too long to respond."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def completion(self, 
                  prompt: str, 
                  model: str = "ggml-org/gemma-3-4b-it-GGUF",
                  **kwargs) -> str:
        """
        Send a completion request (OpenAI-compatible)
        
        Args:
            prompt: Text prompt for completion
            model: Model name
            **kwargs: Additional parameters
        
        Returns:
            Generated response text
        """
        # Merge default options with provided kwargs
        options = {**self.default_options, **kwargs}
        
        data = {
            "model": model,
            "prompt": prompt,
            **options
        }
        
        try:
            response = requests.post(
                self.completions_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("text", "").strip()
                else:
                    return "No response generated"
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to LLaMA server. Make sure the Podman container is running."
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be taking too long to respond."
        except Exception as e:
            return f"Error: {str(e)}"

    def ask_with_context(self, query: str, context: str = "", **kwargs) -> str:
        """
        Ask a question with context (Ollama-compatible interface)
        
        Args:
            query: User question
            context: Retrieved context from RAG
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        # Create messages for chat completion
        messages = [
            {"role": "system", "content": INITIAL_PROMPT},
        ]
        
        if context.strip():
            messages.append({
                "role": "system", 
                "content": f"Context: {context}"
            })
        
        messages.append({
            "role": "user", 
            "content": query
        })
        
        return self.chat_completion(messages, **kwargs)
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        try:
            # Try to get models as a health check
            models = self.get_models()
            health_check = self.test_connection()
            
            return {
                "status": "healthy" if health_check else "unhealthy",
                "base_url": self.base_url,
                "models": models,
                "endpoints": {
                    "chat": self.chat_url,
                    "completions": self.completions_url,
                    "health": self.health_url,
                    "models": self.models_url
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "base_url": self.base_url
            }

def main():
    """Test the LLaMA client"""
    print("LLaMA Server API Client Test")
    print("=" * 50)
    
    client = LlamaClient()
    
    # Test connection
    print("Testing connection...")
    if client.test_connection():
        print("✓ Server is responding")
    else:
        print("✗ Server is not responding")
        return
    
    # Get server info
    print("\nServer information:")
    info = client.get_server_info()
    print(json.dumps(info, indent=2))
    
    # Test chat completion
    print("\nTesting chat completion...")
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you tell me about the Jetson Nano?"}
    ]
    
    response = client.chat_completion(test_messages)
    print(f"Response: {response}")
    
    # Test with context (RAG-style)
    print("\nTesting RAG-style query...")
    context = "The Jetson Nano is a compact, powerful computer designed by NVIDIA for AI applications at the edge."
    query = "What is the Jetson Nano?"
    
    response = client.ask_with_context(query, context)
    print(f"RAG Response: {response}")

if __name__ == "__main__":
    main()