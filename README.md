# 🚚 Logistics Document Intelligence Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg?color=ff4b4b)](https://logistic-docs-intelligent-assistant.streamlit.app/)

An AI-powered POC system designed for Transportation Management Systems (TMS). Upload logistics documents (PDF, DOCX, TXT) and interact with them using natural language. The system provides grounded answers, structured data extraction, and confidence scoring.

## 🚀 Key Features

- **Document Ingestion**: Seamlessly process Rate Confirmations, BOLs, Invoices, and more.
- **RAG-based Q&A**: Ask natural language questions and get answers grounded in your documents.
- **Structured Extraction**: Automatically extract 15+ logistics fields (Reference IDs, Carrier info, Rates, etc.) into JSON.
- **Hallucination Guardrails**: Multi-layered checks to ensure answer accuracy.
- **Confidence Scoring**: Real-time reliability metrics for every response.

---

## 🏗️ Architecture & Tech Stack

The system follows a modern RAG (Retrieval-Augmented Generation) architecture:

- **Frontend**: [Streamlit](https://streamlit.io/) for a lightweight, interactive UI.
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) for a high-performance asynchronous API.
- **LLM**: [Groq](https://groq.com/) (LLaMA-3) for ultra-fast, deterministic inference.
- **Orchestration**: [LangChain](https://www.langchain.com/) for RAG pipelines and LCEL chains.
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) for efficient local similarity search.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.

---

## 🛠️ Technical Implementation

### 1. Chunking Strategy
We use a **Semantic Section-Based Chunking** approach:
- **Header Recognition**: Identifies common logistics sections (Carrier Details, Rate Breakdown, etc.) using regex.
- **Title-Based Grouping**: Related sections are grouped together before splitting to preserve semantic context.
- **Recursive Splitting**: Standard `RecursiveCharacterTextSplitter` is applied to grouped sections with strategic overlap.

### 2. Retrieval Method
- **MMR (Maximal Marginal Relevance)**: Used to balance relevance and diversity, ensuring the LLM sees a broad context of the document.
- **Top-K**: Retrieves the most relevant $K$ chunks for grounded answering.

### 3. Guardrails & Grounding
- **Safety Filter**: Blocks unsafe or irrelevant queries.
- **Grounding Prompt**: Hardened system instructions that force the LLM to only answer from provided context.
- **Forbidden Phrases**: Penalizes transparency phrases (e.g., "generally", "typically") that indicate hallucination.

### 4. Confidence Scoring
Heuristic-based scoring (0.0 to 1.0) based on:
- **Retrieval Quality**: Number of relevant chunks found.
- **Answer Length & Coverage**: Penalizes extremely short or vague answers.
- **Grounding Check**: Significant penalty if "cannot find the answer" is detected.

### 5. Structured Extraction
- **Pydantic Models**: Strict schema validation for 20+ logistics fields.
- **Auto-Indexing**: Extracted data is formatted as text and re-indexed into the vector store, making structured data searchable via RAG.

---

## 📂 Project Structure

```text
.
├── backend/
│   ├── main.py                 # FastAPI Entry point
│   └── src/
│       ├── api/routes.py       # API Endpoints (/upload, /ask, /extract)
│       ├── core/config.py      # App configurations
│       ├── models/             # Pydantic schemas (QA & Extraction)
│       └── services/           # Core Logic
│           ├── ingestion.py    # Document parsing & chunking
│           ├── rag.py          # RAG pipeline & Guardrails
│           ├── extraction.py   # LLM-based structured extraction
│           └── vector_store.py # FAISS & Embeddings management
├── frontend/
│   └── app.py                  # Streamlit UI
├── data/                       # sample test data
├── docs/                       # Project Requirement Document
├── run_local.sh                # Automation script for local setup
└── Makefile                    # Development shortcuts
```
Alternatively, use the `Makefile`:
```bash
make build	    	    #Builds Docker containers using Docker Compose.
make run-docker	    	#Starts the full application stack using Docker Compose.
make run-local	    	#Runs the application locally using the run_local.sh script.
```

---

## 💻 Local Setup

### Prerequisites
- Python 3.11.5+
- Groq API Key

### Linux / Mac / WSL
The easiest way to run the project is using the provided shell script:

```bash
# 1. Setup environment variables (Add your GROQ_API_KEY)
cp .env.example .env
# Also copy to backend/ and frontend/ directories for reliability
cp .env backend/.env
cp .env frontend/.env

# 2. Run the application
chmod +x run_local.sh
./run_local.sh
```

### Windows (PowerShell)
For native Windows users without WSL, follow these manual steps:

```powershell
# 1. Setup environment variables
copy .env.example .env

# 2. Setup virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Backend (In a separate terminal)
# Note: Ensure you are in the project root
$env:PYTHONPATH = "$(Get-Location)\backend;$(Get-Location)\backend\src"
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 5. Run Frontend (In another terminal)
streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
```

### Run with Docker

If you prefer using Docker, you can spin up the entire stack (Backend + Frontend + Log Viewer) using Docker Compose:

```bash
# 1. Build and start the containers
docker-compose up --build

# 2. Access the services:
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8000
# - Dozzle (Log Viewer): http://localhost:8888
```

---

## ⚠️ Known Limitations & Future Improvements

- **Failure Cases**: Highly complex table structures in PDFs may occasionally lead to misaligned extraction.
- **Improvements**:
  - Implement OCR (Tesseract/PaddleOCR) for scanned images.
  - Add Persistent Vector DB (ChromaDB/Pinecone) for long-term storage.
  - Multi-agent extraction for higher accuracy in complex invoices.

---

## ☁️ Standalone App & Streamlit Cloud Deployment

For quick deployment or testing without the FastAPI backend, use the standalone `streamlit_app.py`. This file contains all the necessary backend logic (ingestion, RAG, extraction) inside a single file with **zero internal dependencies**.

### Why use this?
- **Streamlit Cloud**: Deploy directly to [Streamlit Community Cloud](https://share.streamlit.io/) by just pointing to this file.
- **Portability**: This single file provides the full functionality of the project.

### How to run:
```bash
# Set your API key
export GROQ_API_KEY="your_key_here"

# Run the standalone app
streamlit run streamlit_app.py
```

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
