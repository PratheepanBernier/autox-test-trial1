import os
import sys
from dotenv import load_dotenv

# Add current dir to path
sys.path.append(os.getcwd())

load_dotenv()

def test_standalone_services():
    print("Testing standalone streamlit_app.py services...")
    try:
        from streamlit_app import DocumentIngestionService, VectorStoreService, RAGService, ExtractionService
        
        ingest = DocumentIngestionService()
        print("✓ DocumentIngestionService initialized")
        
        vs = VectorStoreService()
        print("✓ VectorStoreService initialized")
        
        rag = RAGService(vs)
        print("✓ RAGService initialized")
        
        extractor = ExtractionService()
        print("✓ ExtractionService initialized")
        
        print("\nAll standalone services initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize standalone services: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_standalone_services()
