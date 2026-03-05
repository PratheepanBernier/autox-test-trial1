import os
import sys
from dotenv import load_dotenv

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), "backend", "src"))

load_dotenv()

def test_backend_services():
    print("Testing multi-service backend services...")
    try:
        from services.ingestion import ingestion_service
        from services.vector_store import vector_store_service
        from services.rag import rag_service
        from services.extraction import extraction_service
        
        print("✓ ingestion_service imported")
        print("✓ vector_store_service imported")
        print("✓ rag_service imported")
        print("✓ extraction_service imported")
        
        # Test initialization
        print(f"Embedding model: {vector_store_service.embeddings.model_name}")
        
        print("\nAll backend services initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize backend services: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_backend_services()
