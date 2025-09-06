import whisper, requests, os, sounddevice as sd, numpy as np, tempfile, wave
import faiss
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from config import *

# Load sentence transformer model for document embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a smaller, more accessible model

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model(WHISPER_CONFIG["model"])

# Current directory and path for beep sound files
current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "../Gemma2/assets/bip.wav")
bip2_sound = os.path.join(current_dir, "../Gemma2/assets/bip2.wav")

# PDF Document Processor optimized for Jetson Orin Nano
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
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that don't add semantic value
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks optimized for semantic search"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            
            # Only add chunks with meaningful content
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

# Enhanced Vector Database with persistence and PDF support
class VectorDatabase:
    def __init__(self, dim: int, cache_dir: str = "./rag_cache"):
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create FAISS index optimized for Jetson (using IndexFlatL2 for accuracy)
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
        self.document_metadata = []  # Store source info for each chunk
        
        # PDF processor
        self.pdf_processor = PDFProcessor(
            chunk_size=PDF_CONFIG.get("chunk_size", 500),
            chunk_overlap=PDF_CONFIG.get("chunk_overlap", 50)
        )
        
        # Load existing index if available
        self._load_cache()
    
    def _get_cache_path(self, name: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{name}.pkl"
    
    def _save_cache(self):
        """Save index and documents to cache for fast loading"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.cache_dir / "faiss_index.bin"))
            
            # Save documents and metadata
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
        """Load existing index and documents from cache"""
        try:
            index_path = self.cache_dir / "faiss_index.bin"
            docs_path = self._get_cache_path("documents")
            
            if index_path.exists() and docs_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load documents and metadata
                with open(docs_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.documents = cache_data['documents']
                    self.document_metadata = cache_data['metadata']
                
                print(f"RAG cache loaded: {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Reset to empty state on error
            self.index = faiss.IndexFlatL2(self.dim)
            self.documents = []
            self.document_metadata = []
    
    def add_documents(self, docs: List[str], source: str = "manual"):
        """Add documents with metadata"""
        if not docs:
            return
        
        # Generate embeddings in batches for memory efficiency
        batch_size = 32  # Optimized for Jetson memory
        
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            embeddings = embedding_model.encode(batch_docs, show_progress_bar=False)
            
            # Add to FAISS index
            self.index.add(np.array(embeddings, dtype=np.float32))
            
            # Store documents and metadata
            self.documents.extend(batch_docs)
            self.document_metadata.extend([source] * len(batch_docs))
        
        # Save cache after adding documents
        self._save_cache()
    
    def add_pdf(self, pdf_path: str) -> bool:
        """Process and add a PDF to the knowledge base"""
        try:
            # Check if PDF already processed
            pdf_hash = self._get_file_hash(pdf_path)
            hash_path = self._get_cache_path(f"pdf_{pdf_hash}")
            
            if hash_path.exists():
                print(f"PDF {pdf_path} already processed")
                return True
            
            # Process PDF
            chunks = self.pdf_processor.process_pdf(pdf_path)
            if not chunks:
                return False
            
            # Add chunks to vector database
            self.add_documents(chunks, source=f"pdf:{Path(pdf_path).name}")
            
            # Mark PDF as processed
            with open(hash_path, 'w') as f:
                f.write(f"Processed: {pdf_path}")
            
            print(f"Successfully added PDF: {pdf_path} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return False
    
    def add_pdf_directory(self, pdf_dir: str) -> int:
        """Process all PDFs in a directory"""
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            print(f"Directory {pdf_dir} does not exist")
            return 0
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        successful = 0
        
        for pdf_path in pdf_files:
            if self.add_pdf(str(pdf_path)):
                successful += 1
        
        print(f"Processed {successful}/{len(pdf_files)} PDFs from {pdf_dir}")
        return successful
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to track processing"""
        return hashlib.md5(str(file_path).encode() + str(Path(file_path).stat().st_mtime).encode()).hexdigest()
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Enhanced search with metadata"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Safety check
                results.append({
                    'text': self.documents[idx],
                    'source': self.document_metadata[idx],
                    'score': float(distances[0][i])
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'sources': list(set(self.document_metadata)),
            'cache_dir': str(self.cache_dir)
        }

# Create a VectorDatabase and add documents to it
db = VectorDatabase(dim=FAISS_CONFIG["dimension"], cache_dir=PDF_CONFIG["cache_directory"])
db.add_documents(KNOWLEDGE_DOCS, source="config")

# Auto-load PDFs from default directory if it exists
if os.path.exists(PDF_CONFIG["pdf_directory"]):
    print(f"Loading PDFs from {PDF_CONFIG['pdf_directory']}...")
    loaded_count = db.add_pdf_directory(PDF_CONFIG["pdf_directory"])
    if loaded_count > 0:
        print(f"Loaded {loaded_count} PDFs into knowledge base")
else:
    print(f"PDF directory {PDF_CONFIG['pdf_directory']} not found. Create it and add PDFs for enhanced RAG.")

# Find the device for audio recording by matching part of the device name
def find_device(device_name_substring):
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_inputs'] > 0 and device_name_substring.lower() in device['name'].lower():
                return i
        # If specific device not found, use first available input device
        for i, device in enumerate(devices):
            if device['max_inputs'] > 0:
                return i
        # Fallback to default
        return sd.default.device[0]
    except:
        return None

# Play sound (beep) to signal recording start/stop
def play_sound(sound_file):
    if os.path.exists(sound_file):
        # Cross-platform sound playback
        if os.name == 'posix':  # Linux/macOS
            if os.path.exists('/usr/bin/aplay'):  # Linux
                os.system(f"aplay {sound_file}")
            else:  # macOS
                os.system(f"afplay {sound_file}")
        else:  # Windows
            os.system(f"start {sound_file}")
    else:
        print("Beep!")  # Fallback if sound file doesn't exist

# Record audio using sounddevice, save it as a .wav file
def record_audio(filename, duration=AUDIO_CONFIG["duration"], fs=AUDIO_CONFIG["sample_rate"]):
    try:
        device_id = find_device("920")  # Try to find Logitech 920
        if device_id is not None:
            sd.default.device = device_id
    except:
        pass  # Use default device if not found
    
    play_sound(bip_sound)  # Start beep
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to complete
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    play_sound(bip2_sound)  # End beep

# Transcribe recorded audio to text using Whisper
def transcribe_audio(filename):
    return whisper_model.transcribe(filename, language="en")['text']

# Send a query and context to Ollama server for completion
def ask_ollama(query, context):
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

# Generate a response using enhanced Retrieval-Augmented Generation (RAG)
def rag_ask(query):
    # Search for related docs in the FAISS index with metadata
    search_results = db.search(query, top_k=FAISS_CONFIG["top_k"])
    
    if not search_results:
        context = "No relevant documents found in knowledge base."
    else:
        # Create context with source attribution for better responses
        context_parts = []
        for result in search_results:
            context_parts.append(result['text'])
        context = " ".join(context_parts)
    
    return ask_ollama(query, context)  # Ask Ollama using the retrieved context

# Enhanced PDF management functions
def add_pdf_to_knowledge_base(pdf_path: str) -> bool:
    """Add a single PDF to the knowledge base"""
    return db.add_pdf(pdf_path)

def load_pdfs_from_directory(pdf_dir: str = None) -> int:
    """Load all PDFs from a directory"""
    if pdf_dir is None:
        pdf_dir = PDF_CONFIG["pdf_directory"]
    return db.add_pdf_directory(pdf_dir)

def get_rag_stats() -> Dict:
    """Get current RAG database statistics"""
    return db.get_stats()

def clear_rag_cache():
    """Clear the RAG cache (useful for testing)"""
    import shutil
    cache_dir = PDF_CONFIG["cache_directory"]
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"RAG cache cleared: {cache_dir}")
    
    # Reinitialize database
    global db
    db = VectorDatabase(dim=FAISS_CONFIG["dimension"], cache_dir=cache_dir)
    db.add_documents(KNOWLEDGE_DOCS, source="config")
    print("RAG database reinitialized")

# Convert text to speech using system TTS
def text_to_speech(text):
    # Cross-platform TTS
    if os.name == 'posix':  # Linux/macOS
        if os.path.exists('/usr/bin/espeak'):  # Linux with espeak
            os.system(f'espeak "{text}"')
        elif os.path.exists('/usr/bin/say'):  # macOS
            os.system(f'say "{text}"')
        else:  # Try piper if available (Jetson)
            piper_path = "/home/asier/piper/build/piper"
            if os.path.exists(piper_path):
                os.system(f'echo "{text}" | {piper_path} --model /usr/local/share/piper/models/en-us-lessac-medium.onnx --output_file response.wav && aplay response.wav')
            else:
                print(f"🤖 Assistant: {text}")  # Fallback to text output
    else:  # Windows
        os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')

# Main loop for the assistant
def main():
    print("Voice Assistant with Ollama + Gemma3n")
    print("Optimized for Jetson Orin Nano")
    print("Press Ctrl+C to exit")
    print("-" * 50)
    
    while True:
        try:
            # Create a temporary .wav file for the recording
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                record_audio(tmpfile.name)  # Record the audio input
                transcribed_text = transcribe_audio(tmpfile.name)  # Convert speech to text
                print(f"You said: {transcribed_text}")
                
                if transcribed_text.strip():  # Only process if there's actual text
                    response = rag_ask(transcribed_text)  # Generate response using RAG and Ollama
                    print(f"Assistant: {response}")
                    if response and not response.startswith("Error"):
                        text_to_speech(response)  # Convert response to speech
                else:
                    print("Assistant: I didn't hear anything. Please try again.")
                
                # Clean up temporary file
                os.unlink(tmpfile.name)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

# Entry point of the script
if __name__ == "__main__":
    main() 