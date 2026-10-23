import sys
import os

# Adds project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embeddings import GeminiDocumentEmbedder

# Tests document retrieval
embedder = GeminiDocumentEmbedder()
query_result = embedder.query_documents('engineering requirements', n_results=3)

print('Sample query results:')
print(f'Query: {query_result.get("query", "Unknown")}')
print(f'Found {query_result.get("count", 0)} documents\n')

if "error" in query_result:
    print(f"Error: {query_result['error']}")
else:
    # Extract  results from ChromaDB format
    chroma_results = query_result.get("results", {})
    documents = chroma_results.get("documents", [[]])[0]  # First query results
    metadatas = chroma_results.get("metadatas", [[]])[0]  # First query metadata
    distances = chroma_results.get("distances", [[]])[0]  # First query distances
    
    print("="*60)
    
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        # Extract useful info from metadata
        doc_type = metadata.get('type', 'unknown')
        department = metadata.get('department', 'Unknown')
        
        if doc_type == 'course':
            identifier = metadata.get('course_name', 'Unknown Course')
        elif doc_type == 'program':
            identifier = metadata.get('program', 'Unknown Program')
        elif doc_type == 'department':
            identifier = metadata.get('department', 'Unknown Department')
        else:
            identifier = f"Document {i+1}"
        
        # Preview the content
        preview = doc[:200] + "..." if len(doc) > 200 else doc
        
        print(f'{i+1}. {doc_type.title()}: {identifier}')
        print(f'   Department: {department}')
        print(f'   Similarity Score: {1-distance:.3f}')  # Convert distance to similarity
        print(f'   Preview: {preview}')
        print('-' * 60)

print(f"\nSuccessfully retrieved {len(documents)} relevant UCSB Engineering documents!")