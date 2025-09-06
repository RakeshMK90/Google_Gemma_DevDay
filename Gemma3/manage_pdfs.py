#!/usr/bin/env python3
"""
PDF Management Utility for RAG Knowledge Base
Optimized for Jetson Orin Nano
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from assistant_ollama import (
    add_pdf_to_knowledge_base, load_pdfs_from_directory,
    get_rag_stats, clear_rag_cache
)

def add_pdf(pdf_path: str):
    """Add a single PDF to the knowledge base"""
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return False
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: {pdf_path} is not a PDF file")
        return False
    
    print(f"Adding PDF: {pdf_path}")
    success = add_pdf_to_knowledge_base(pdf_path)
    
    if success:
        print(f"✓ Successfully added {pdf_path}")
    else:
        print(f"✗ Failed to add {pdf_path}")
    
    return success

def add_directory(pdf_dir: str):
    """Add all PDFs from a directory"""
    if not os.path.exists(pdf_dir):
        print(f"Error: Directory {pdf_dir} does not exist")
        return False
    
    print(f"Loading PDFs from directory: {pdf_dir}")
    count = load_pdfs_from_directory(pdf_dir)
    
    if count > 0:
        print(f"✓ Successfully loaded {count} PDFs")
    else:
        print("No PDFs found or processed")
    
    return count > 0

def show_stats():
    """Display knowledge base statistics"""
    stats = get_rag_stats()
    
    print("\nKnowledge Base Statistics:")
    print("=" * 40)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Index size: {stats['index_size']}")
    print(f"Cache directory: {stats['cache_dir']}")
    
    if stats['sources']:
        print(f"\nSources ({len(stats['sources'])}):")
        for source in sorted(stats['sources']):
            print(f"  - {source}")
    else:
        print("\nNo sources found")

def clear_cache():
    """Clear the RAG cache"""
    print("Clearing RAG cache...")
    clear_rag_cache()
    print("✓ Cache cleared and database reinitialized")

def setup_pdf_directory():
    """Setup the default PDF directory"""
    pdf_dir = Path(PDF_CONFIG["pdf_directory"])
    pdf_dir.mkdir(exist_ok=True)
    
    readme_file = pdf_dir / "README.txt"
    with open(readme_file, 'w') as f:
        f.write("""PDF Knowledge Base Directory

This directory is used to store PDF files for the RAG (Retrieval-Augmented Generation) system.

Usage:
1. Place PDF files in this directory
2. Run: python manage_pdfs.py --add-directory ./pdfs
3. Or use: python manage_pdfs.py --add-file path/to/specific.pdf

The assistant will automatically index these PDFs and use them to answer questions.

Configuration:
- Chunk size: {chunk_size} words
- Chunk overlap: {chunk_overlap} words  
- Max file size: {max_file_size_mb} MB
- Cache directory: {cache_directory}

For automatic loading, PDFs in this directory are processed when the assistant starts.
""".format(**PDF_CONFIG))
    
    print(f"✓ PDF directory setup complete: {pdf_dir}")
    print(f"  README file created: {readme_file}")

def main():
    parser = argparse.ArgumentParser(
        description="PDF Management for RAG Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_pdfs.py --stats                    # Show current stats
  python manage_pdfs.py --add-file document.pdf    # Add single PDF
  python manage_pdfs.py --add-directory ./pdfs     # Add all PDFs from directory
  python manage_pdfs.py --setup                    # Setup PDF directory
  python manage_pdfs.py --clear                    # Clear cache
        """
    )
    
    parser.add_argument('--add-file', metavar='PDF_PATH', 
                       help='Add a single PDF file to knowledge base')
    parser.add_argument('--add-directory', metavar='DIR_PATH', 
                       help='Add all PDFs from directory to knowledge base')
    parser.add_argument('--stats', action='store_true',
                       help='Show knowledge base statistics')
    parser.add_argument('--clear', action='store_true',
                       help='Clear the RAG cache and reinitialize')
    parser.add_argument('--setup', action='store_true',
                       help='Setup the default PDF directory')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    try:
        if args.setup:
            setup_pdf_directory()
        
        if args.add_file:
            add_pdf(args.add_file)
        
        if args.add_directory:
            add_directory(args.add_directory)
        
        if args.clear:
            clear_cache()
        
        if args.stats:
            show_stats()
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()