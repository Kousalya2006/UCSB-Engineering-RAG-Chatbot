import json
import os
from typing import List, Dict, Any
from src.config.settings import Config

class DataValidator:
    """Validate and preview data before embedding (no API calls)"""
    
    def __init__(self, processed_file: str = None):
        self.config = Config()
        # Use config for file path
        self.processed_file = processed_file or os.path.join(self.config.DATA_PATH, "processed_documents.json")
    
    def validate_data_structure(self) -> bool:
        """Validate that processed data has correct structure"""
        if not os.path.exists(self.processed_file):
            print(f"File not found: {self.processed_file}")
            print(f"   Run: python -m src.data.data_processor first")
            return False
        
        try:
            with open(self.processed_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            print(f"File found: {self.processed_file}")
            print(f"Total documents: {len(documents)}")
            
            # Check structure
            required_fields = ['doc_id', 'content', 'metadata']
            for i, doc in enumerate(documents[:5]):  # Check first 5
                for field in required_fields:
                    if field not in doc:
                        print(f"Document {i} missing field: {field}")
                        return False
            
            print("Document structure valid")
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def analyze_content(self):
        """Analyze content without making API calls"""
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Content analysis
        total_chars = sum(len(doc['content']) for doc in documents)
        if documents:
            avg_chars = total_chars / len(documents)
        else:
            avg_chars = 0
        
        # Type breakdown
        types = {}
        for doc in documents:
            doc_type = doc['metadata']['type']
            types[doc_type] = types.get(doc_type, 0) + 1
        
        # Department breakdown
        departments = {}
        for doc in documents:
            dept = doc['metadata'].get('department', 'Unknown')
            departments[dept] = departments.get(dept, 0) + 1
        
        print(f"\nCONTENT ANALYSIS")
        print(f"Total characters: {total_chars:,}")
        print(f"Average chars per document: {avg_chars:.1f}")
        print(f"Longest document: {max(len(doc['content']) for doc in documents):,} chars")
        print(f"Shortest document: {min(len(doc['content']) for doc in documents):,} chars")
        
        print(f"\nDOCUMENT TYPES:")
        for doc_type, count in types.items():
            print(f"  {doc_type}: {count} documents")
        
        print(f"\nDEPARTMENTS:")
        for dept, count in departments.items():
            print(f"  {dept}: {count} documents")
    
    def estimate_embedding_cost(self):
        """Estimate API calls needed (no actual calls made)"""
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Use config for embedding limits
        chunk_size = 900  # Conservative estimate for embedding chunks
        
        # Simulate chunking logic
        total_chunks = 0
        for doc in documents:
            content = doc['content']
            if len(content) <= chunk_size:
                chunks = 1
            else:
                # Rough chunking estimate
                chunks = len(content) // chunk_size + 1
            total_chunks += chunks
        
        print(f"\nEMBEDDING COST ESTIMATE")
        print(f"Total documents: {len(documents)}")
        print(f"Estimated chunks: {total_chunks}")
        print(f"API calls needed: {total_chunks}")
        print(f"Using model: {self.config.EMBEDDING_MODEL}")
        print(f"Gemini free tier: 1,500 requests/day")
        
        if total_chunks <= 1000:
            print("Should be FREE on Gemini free tier")
        elif total_chunks <= 1500:
            print("Close to free tier limit, should still be FREE")
        else:
            print("Might exceed free tier, consider splitting into batches")
    
    def preview_sample_documents(self, n: int = 5):
        """Preview sample documents"""
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"\nSAMPLE DOCUMENTS (first {n}):")
        for i, doc in enumerate(documents[:n]):
            print(f"\n--- Document {i+1} ---")
            print(f"ID: {doc['doc_id']}")
            print(f"Type: {doc['metadata']['type']}")
            print(f"Department: {doc['metadata'].get('department', 'N/A')}")
            print(f"Content length: {len(doc['content'])} chars")
            print(f"Content preview:")
            print(f"  {doc['content'][:200]}...")
            if len(doc['content']) > 200:
                print(f"  ... and {len(doc['content']) - 200} more characters")
    
    def check_for_potential_issues(self):
        """Check for potential embedding issues"""
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        issues = []
        
        # Check for very long documents
        for doc in documents:
            if len(doc['content']) > 5000:
                issues.append(f"Long document: {doc['doc_id']} ({len(doc['content'])} chars)")
        
        # Check for very short documents
        for doc in documents:
            if len(doc['content']) < 50:
                issues.append(f"Short document: {doc['doc_id']} ({len(doc['content'])} chars)")
        
        # Check for empty content
        for doc in documents:
            if not doc['content'].strip():
                issues.append(f"Empty document: {doc['doc_id']}")
        
        if issues:
            print(f"\nPOTENTIAL ISSUES:")
            for issue in issues[:10]:  # Show first 10
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        else:
            print(f"\nNo major issues found")
    
    def check_embedding_readiness(self) -> bool:
        """Check if data is ready for embedding"""
        try:
            # Validate config
            is_valid, error_msg = self.config.validate_config()
            if not is_valid:
                print(f"Config error: {error_msg}")
                return False
            
            # Check if processed data exists
            if not self.validate_data_structure():
                return False
            
            # Check for critical issues
            with open(self.processed_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            if len(documents) == 0:
                print("No documents found")
                return False
            
            # Check for empty documents
            empty_docs = [doc for doc in documents if not doc['content'].strip()]
            if empty_docs:
                print(f"Found {len(empty_docs)} empty documents")
                return False
            
            print("Data is ready for embedding!")
            return True
            
        except Exception as e:
            print(f"Error checking embedding readiness: {e}")
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        print("VALIDATING DATA FOR EMBEDDING")
        print("=" * 50)
        
        if not self.validate_data_structure():
            return False
        
        self.analyze_content()
        self.estimate_embedding_cost()
        self.preview_sample_documents()
        self.check_for_potential_issues()
        
        ready = self.check_embedding_readiness()
        
        print("\n" + "=" * 50)
        if ready:
            print("VALIDATION COMPLETE - Ready to run embeddings!")
            print(f"Next step: python -m src.core.embeddings")
        else:
            print("VALIDATION FAILED - Fix issues before embedding")
        
        return ready

def main():
    """Main function for direct execution"""
    validator = DataValidator()
    success = validator.run_full_validation()
    
    if success:
        print(f"\nReady to proceed with embedding!")
    else:
        print(f"\nPlease fix validation issues first")

# Usage
if __name__ == "__main__":
    main()