"""
Example: Basic Document Extraction

This example demonstrates how to use the ADE MCP server for basic document extraction.
"""

import base64
import json
from pathlib import Path

def encode_file_to_base64(file_path):
    """Encode a file to base64 string."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def main():
    # Example 1: Extract from local file
    print("Example 1: Extract from local file")
    print("-" * 40)
    
    # In Claude, you would use:
    # result = await ade_extract_from_path(path="/path/to/document.pdf")
    
    file_path = "/path/to/your/document.pdf"
    print(f"Extracting from: {file_path}")
    print()
    
    # Example 2: Extract from URL
    print("Example 2: Extract from URL")
    print("-" * 40)
    
    # In Claude, you would use:
    # result = await ade_extract_from_url(url="https://example.com/document.pdf")
    
    url = "https://example.com/document.pdf"
    print(f"Extracting from URL: {url}")
    print()
    
    # Example 3: Batch processing
    print("Example 3: Batch processing")
    print("-" * 40)
    
    # In Claude, you would use:
    # result = await ade_extract_batch(file_paths=["/path/to/doc1.pdf", "/path/to/doc2.pdf"])
    
    file_paths = [
        "/path/to/doc1.pdf",
        "/path/to/doc2.pdf",
        "/path/to/doc3.pdf"
    ]
    print(f"Processing {len(file_paths)} documents in batch")
    print()
    
    # Example 4: Extract from base64-encoded document
    print("Example 4: Extract from base64-encoded document")
    print("-" * 40)
    
    # Encode a file to base64
    # pdf_base64 = encode_file_to_base64("/path/to/document.pdf")
    
    # In Claude, you would use:
    # result = await ade_extract_raw_chunks(pdf_base64=pdf_base64)
    
    print("Document encoded and ready for extraction")

if __name__ == "__main__":
    main()