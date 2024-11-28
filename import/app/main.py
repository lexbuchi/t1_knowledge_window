# main.py

import argparse
import asyncio
from process_document import process_document
import os
import sys

def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    return markdown_text

def main():
    parser = argparse.ArgumentParser(description='Process a markdown file into the knowledge graph.')
    parser.add_argument('file', help='Path to the markdown (.md) file to process')
    parser.add_argument('--document_name', help='Name of the document', default=None)
    parser.add_argument('--chunk_size', type=int, default=250, help='Chunk size for text splitting')
    parser.add_argument('--chunk_overlap', type=int, default=30, help='Chunk overlap for text splitting')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        sys.exit(1)

    markdown_text = read_markdown_file(file_path)

    # Use the file name as document name if not provided
    document_name = args.document_name or os.path.splitext(os.path.basename(file_path))[0]

    # Run the processing asynchronously
    asyncio.run(process_document(markdown_text, document_name, args.chunk_size, args.chunk_overlap))

if __name__ == '__main__':
    main()