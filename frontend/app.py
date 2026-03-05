import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Logistics Doc Intelligence", layout="wide")

st.title("🚚 Logistics Document Intelligence Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Allow users to input their own backend URL
    default_backend = os.getenv("BACKEND_URL", "http://localhost:8000")
    BACKEND_URL = st.text_input(
        "Backend API URL",
        value=default_backend,
        help="The URL of the FastAPI backend."
    )
    
    st.divider()
    st.markdown("""
    ### About
    This assistant helps you interact with logistics documents using RAG (Retrieval-Augmented Generation).
    
    **Features:**
    - Document Q&A
    - Structured Data Extraction
    - Confidence Scoring
    - Grounded Answers
    """)

# Helper to check backend health
def check_backend():
    try:
        response = requests.get(f"{BACKEND_URL}/ping")
        return response.status_code == 200
    except:
        return False

# Check if backend is reachable
is_backend_up = check_backend()

if not is_backend_up:
    st.error(f"⚠️ Cannot connect to backend at {BACKEND_URL}. Please ensure the backend is running.")
    if st.button("Retry Connection"):
        st.rerun()
    st.stop()

# Main tabs
tabs = st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])

with tabs[0]:
    st.header("Upload Logistics Documents")
    st.write("Upload PDF, DOCX, or TXT documents to index them for Q&A.")
    
    uploaded_files = st.file_uploader(
        "Choose documents", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Documents"):
        if uploaded_files:
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type)) 
                for f in uploaded_files
            ]
            
            with st.spinner("Processing documents (chunking, embedding, indexing)..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files_to_send
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(result["message"])
                        
                        if result.get("extractions"):
                            with st.expander("Extraction Summary"):
                                for ext in result["extractions"]:
                                    status = "✅" if ext["structured_data_extracted"] else "❌"
                                    st.write(f"{status} **{ext['filename']}**: {ext['text_chunks']} chunks")
                        
                        if result.get("errors"):
                            for err in result["errors"]:
                                st.warning(err)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
        else:
            st.warning("Please upload at least one document.")

with tabs[1]:
    st.header("Question Answering")
    st.write("Ask questions about your uploaded documents.")
    
    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for src in message["sources"]:
                        st.caption(f"Source: {src['metadata']['source']}")
                        st.write(src["text"])

    # Chat input
    if prompt := st.chat_input("What is the agreed rate for this shipment?"):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"question": prompt, "chat_history": []}
                    response = requests.post(f"{BACKEND_URL}/ask", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        confidence = data["confidence_score"]
                        sources = data["sources"]
                        
                        st.markdown(answer)
                        st.caption(f"Confidence Score: {confidence:.2f}")
                        
                        if sources:
                            with st.expander("View Sources"):
                                for src in sources:
                                    st.caption(f"Source: {src['metadata']['source']}")
                                    st.write(src["text"])
                        
                        # Save assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

with tabs[2]:
    st.header("Structured Data Extraction")
    st.write("Extract specific fields from a logistics document.")
    
    extract_file = st.file_uploader(
        "Upload a single document for extraction", 
        type=["pdf", "docx", "txt"],
        key="extraction_uploader"
    )
    
    if st.button("Run Extraction"):
        if extract_file:
            with st.spinner("Extracting data..."):
                try:
                    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                    response = requests.post(f"{BACKEND_URL}/extract", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()["data"]
                        st.success("Extraction complete!")
                        
                        # Display data in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("IDs & Parties")
                            st.write(f"**Reference ID:** {data.get('reference_id', 'N/A')}")
                            st.write(f"**Load ID:** {data.get('load_id', 'N/A')}")
                            st.write(f"**Shipper:** {data.get('shipper', 'N/A')}")
                            st.write(f"**Consignee:** {data.get('consignee', 'N/A')}")
                            
                            st.subheader("Carrier & Driver")
                            carrier = data.get('carrier') or {}
                            st.write(f"**Carrier:** {carrier.get('carrier_name', 'N/A')}")
                            st.write(f"**MC Number:** {carrier.get('mc_number', 'N/A')}")
                            
                            driver = data.get('driver') or {}
                            st.write(f"**Driver:** {driver.get('driver_name', 'N/A')}")
                            st.write(f"**Truck:** {driver.get('truck_number', 'N/A')}")
                        
                        with col2:
                            st.subheader("Stops & Dates")
                            pickup = data.get('pickup') or {}
                            st.write(f"**Pickup:** {pickup.get('city', 'N/A')}, {pickup.get('state', 'N/A')}")
                            st.write(f"**Pickup Date:** {data.get('shipping_date', 'N/A')}")
                            
                            drop = data.get('drop') or {}
                            st.write(f"**Drop:** {drop.get('city', 'N/A')}, {drop.get('state', 'N/A')}")
                            st.write(f"**Delivery Date:** {data.get('delivery_date', 'N/A')}")
                            
                            st.subheader("Rates & Equipment")
                            rate = data.get('rate_info') or {}
                            st.write(f"**Total Rate:** {rate.get('total_rate', 'N/A')} {rate.get('currency', '')}")
                            st.write(f"**Equipment:** {data.get('equipment_type', 'N/A')}")
                        
                        with st.expander("View Full JSON"):
                            st.json(data)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
        else:
            st.warning("Please upload a document.")
