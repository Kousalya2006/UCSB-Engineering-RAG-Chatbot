import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_connection():
    """Test Google Gemini API connection and basic functionality"""
    try:
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Test embeddings
        print("Testing Google Gemini Embeddings...")
        result = genai.embed_content(
            model="models/embedding-001",
            content="Test embedding for UCSB College of Engineering",
            task_type="retrieval_document"
        )
        print(f"Embeddings API working! Embedding dimension: {len(result['embedding'])}")
        
        # Test chat generation
        print("\nTesting Google Gemini Chat...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Hello from UCSB RAG chatbot!'")
        print(f"Chat API working! Response: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"Google Gemini API Error: {e}")
        print("\nSetup Instructions:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Add GOOGLE_API_KEY=your_key_here to your .env file")
        return False

if __name__ == "__main__":
    test_gemini_connection()