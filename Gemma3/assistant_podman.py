#!/usr/bin/env python3
"""
Voice Assistant with Podman/LLaMA Server Integration
Supports both Ollama and Podman deployments based on configuration
Optimized for Jetson Orin Nano
"""

import whisper, os, sounddevice as sd, numpy as np, tempfile, wave
import faiss
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from config import *
from podman_manager import PodmanManager
from llama_client import LlamaClient
import requests
import time

# Load sentence transformer model for document embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model(WHISPER_CONFIG["model"])

# Current directory and path for beep sound files
current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "../Gemma2/assets/bip.wav")
bip2_sound = os.path.join(current_dir, "../Gemma2/assets/bip2.wav")

# PDF Document Processor (reusing from original)
class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with error handling"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better embedding quality"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks optimized for semantic search"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            
            if len(chunk.strip()) > 50:
                chunks.append(self.clean_text(chunk))
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[str]:
        """Complete PDF processing pipeline"""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)
        
        print(f"Processed {pdf_path}: {len(chunks)} chunks extracted")
        return chunks

# Enhanced Vector Database (reusing from original)
class VectorDatabase:
    def __init__(self, dim: int, cache_dir: str = "./rag_cache"):
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
        self.document_metadata = []
        
        self.pdf_processor = PDFProcessor(
            chunk_size=PDF_CONFIG.get("chunk_size", 500),
            chunk_overlap=PDF_CONFIG.get("chunk_overlap", 50)
        )
        
        self._load_cache()
    
    def _get_cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.pkl"
    
    def _save_cache(self):
        try:
            faiss.write_index(self.index, str(self.cache_dir / "faiss_index.bin"))
            
            cache_data = {
                'documents': self.documents,
                'metadata': self.document_metadata
            }
            with open(self._get_cache_path("documents"), 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"RAG cache saved: {len(self.documents)} documents")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _load_cache(self):
        try:
            index_path = self.cache_dir / "faiss_index.bin"
            docs_path = self._get_cache_path("documents")
            
            if index_path.exists() and docs_path.exists():
                self.index = faiss.read_index(str(index_path))
                
                with open(docs_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.documents = cache_data['documents']
                    self.document_metadata = cache_data['metadata']
                
                print(f"RAG cache loaded: {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.index = faiss.IndexFlatL2(self.dim)
            self.documents = []
            self.document_metadata = []
    
    def add_documents(self, docs: List[str], source: str = "manual"):
        if not docs:
            return
        
        batch_size = 32
        
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            embeddings = embedding_model.encode(batch_docs, show_progress_bar=False)
            
            self.index.add(np.array(embeddings, dtype=np.float32))
            
            self.documents.extend(batch_docs)
            self.document_metadata.extend([source] * len(batch_docs))
        
        self._save_cache()
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'source': self.document_metadata[idx],
                    'score': float(distances[0][i])
                })
        
        return results

# Initialize deployment based on configuration
class AIAssistant:
    def __init__(self):
        self.deployment_type = DEPLOYMENT_TYPE
        self.podman_manager = None
        self.llama_client = None
        
        # Initialize vector database
        self.db = VectorDatabase(dim=FAISS_CONFIG["dimension"], cache_dir=PDF_CONFIG["cache_directory"])
        self.db.add_documents(KNOWLEDGE_DOCS, source="config")
        
        # Auto-load PDFs
        if os.path.exists(PDF_CONFIG["pdf_directory"]):
            print(f"Loading PDFs from {PDF_CONFIG['pdf_directory']}...")
            loaded_count = self.db.add_pdf_directory(PDF_CONFIG["pdf_directory"])
            if loaded_count > 0:
                print(f"Loaded {loaded_count} PDFs into knowledge base")
        
        # Initialize deployment
        self._initialize_deployment()
    
    def _initialize_deployment(self):
        """Initialize the chosen deployment type"""
        if self.deployment_type == "podman":
            self._initialize_podman()
        elif self.deployment_type == "ollama":
            self._initialize_ollama()
        else:
            raise ValueError(f"Unknown deployment type: {self.deployment_type}")
    
    def _initialize_podman(self):
        """Initialize Podman deployment"""
        print("Initializing Podman deployment...")
        
        # Create Podman manager
        self.podman_manager = PodmanManager(
            container_name=PODMAN_CONFIG["container_name"],
            port=PODMAN_CONFIG["port"]
        )
        
        # Create LLaMA client
        self.llama_client = LlamaClient(PODMAN_CONFIG["api_url"])
        
        # Auto-start container if configured
        if PODMAN_CONFIG.get("auto_start", True):
            if not self.podman_manager.is_container_running():
                print("Starting Podman container...")
                if self.podman_manager.start_container():
                    print("✓ Container started successfully")
                else:
                    print("✗ Failed to start container")
                    raise RuntimeError("Could not start Podman container")
            else:
                print("✓ Container is already running")
        
        # Test connection
        if not self.llama_client.test_connection():
            print("✗ Failed to connect to LLaMA server")
            raise RuntimeError("Could not connect to LLaMA server")
        else:
            print("✓ Connected to LLaMA server")
    
    def _initialize_ollama(self):
        """Initialize Ollama deployment"""
        print("Initializing Ollama deployment...")
        
        # Test Ollama connection
        try:
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✓ Connected to Ollama server")
            else:
                print("✗ Ollama server error")
                raise RuntimeError("Ollama server not responding")
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to Ollama server")
            print("Make sure to run: ollama serve")
            raise RuntimeError("Ollama server not available")
    
    def ask_model(self, query: str, context: str = "") -> str:
        """Ask the model with context (unified interface)"""
        if self.deployment_type == "podman":
            return self.llama_client.ask_with_context(query, context)
        elif self.deployment_type == "ollama":
            return self._ask_ollama(query, context)
        else:
            return "Error: Unknown deployment type"
    
    def _ask_ollama(self, query: str, context: str) -> str:
        """Legacy Ollama implementation"""
        data = {
            "model": MODEL_NAME,
            "prompt": f"{INITIAL_PROMPT}\nContext: {context}\nQuestion: {query}\nAnswer:",
            "stream": False,
            "options": GENERATION_OPTIONS
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=data, headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                return f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama server. Make sure Ollama is running with 'ollama serve'"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def rag_ask(self, query: str) -> str:
        """Generate a response using enhanced Retrieval-Augmented Generation (RAG)"""
        search_results = self.db.search(query, top_k=FAISS_CONFIG["top_k"])
        
        if not search_results:
            context = "No relevant documents found in knowledge base."
        else:
            context_parts = []
            for result in search_results:
                context_parts.append(result['text'])
            context = " ".join(context_parts)
        
        return self.ask_model(query, context)
    
    def get_deployment_info(self) -> Dict:
        """Get information about current deployment"""
        info = {
            "deployment_type": self.deployment_type,
            "status": "unknown"
        }
        
        if self.deployment_type == "podman":
            info.update({
                "container_name": PODMAN_CONFIG["container_name"],
                "api_url": PODMAN_CONFIG["api_url"],
                "container_running": self.podman_manager.is_container_running() if self.podman_manager else False,
                "server_responding": self.llama_client.test_connection() if self.llama_client else False
            })
            info["status"] = "healthy" if info["container_running"] and info["server_responding"] else "unhealthy"
        
        elif self.deployment_type == "ollama":
            try:
                response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                info["status"] = "healthy" if response.status_code == 200 else "unhealthy"
                info["ollama_url"] = OLLAMA_URL
            except:
                info["status"] = "unhealthy"
        
        return info

# Initialize global assistant instance
assistant = AIAssistant()

# Audio functions (reusing from original)
def find_device(device_name_substring):
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_inputs'] > 0 and device_name_substring.lower() in device['name'].lower():
                return i
        for i, device in enumerate(devices):
            if device['max_inputs'] > 0:
                return i
        return sd.default.device[0]
    except:
        return None

def play_sound(sound_file):
    if os.path.exists(sound_file):
        if os.name == 'posix':
            if os.path.exists('/usr/bin/aplay'):
                os.system(f"aplay {sound_file}")
            else:
                os.system(f"afplay {sound_file}")
        else:
            os.system(f"start {sound_file}")
    else:
        print("Beep!")

def record_audio(filename, duration=AUDIO_CONFIG["duration"], fs=AUDIO_CONFIG["sample_rate"]):
    try:
        device_id = find_device("920")
        if device_id is not None:
            sd.default.device = device_id
    except:
        pass
    
    play_sound(bip_sound)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    play_sound(bip2_sound)

def transcribe_audio(filename):
    return whisper_model.transcribe(filename, language="en")['text']

def text_to_speech(text):
    if os.name == 'posix':
        if os.path.exists('/usr/bin/espeak'):
            os.system(f'espeak "{text}"')
        elif os.path.exists('/usr/bin/say'):
            os.system(f'say "{text}"')
        else:
            piper_path = "/home/asier/piper/build/piper"
            if os.path.exists(piper_path):
                os.system(f'echo "{text}" | {piper_path} --model /usr/local/share/piper/models/en-us-lessac-medium.onnx --output_file response.wav && aplay response.wav')
            else:
                print(f"🤖 Assistant: {text}")
    else:
        os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')

def main():
    print("Voice Assistant with Podman + LLaMA Server")
    print("Optimized for Jetson Orin Nano")
    print("Press Ctrl+C to exit")
    print("-" * 50)
    
    # Show deployment info
    info = assistant.get_deployment_info()
    print(f"Deployment: {info['deployment_type']} ({info['status']})")
    print("-" * 50)
    
    while True:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                record_audio(tmpfile.name)
                transcribed_text = transcribe_audio(tmpfile.name)
                print(f"You said: {transcribed_text}")
                
                if transcribed_text.strip():
                    response = assistant.rag_ask(transcribed_text)
                    print(f"Assistant: {response}")
                    if response and not response.startswith("Error"):
                        text_to_speech(response)
                else:
                    print("Assistant: I didn't hear anything. Please try again.")
                
                os.unlink(tmpfile.name)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()