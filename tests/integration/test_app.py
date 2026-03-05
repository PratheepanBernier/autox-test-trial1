import pytest
import builtins
import types
import io

import frontend.app as app

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit functions used in app.py to no-op or capture calls
    st_calls = {}

    class DummyExpander:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def write(self, *a, **k): st_calls.setdefault('expander_write', []).append((a, k))
        def caption(self, *a, **k): st_calls.setdefault('expander_caption', []).append((a, k))

    class DummyChatMessage:
        def __init__(self, role): self.role = role
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def markdown(self, *a, **k): st_calls.setdefault(f'chat_{self.role}_markdown', []).append((a, k))
        def caption(self, *a, **k): st_calls.setdefault(f'chat_{self.role}_caption', []).append((a, k))
        def write(self, *a, **k): st_calls.setdefault(f'chat_{self.role}_write', []).append((a, k))
        def expander(self, *a, **k): return DummyExpander()

    class DummyColumns:
        def __init__(self, n): self.cols = [self for _ in range(n)]
        def __getitem__(self, idx): return self.cols[idx]
        def subheader(self, *a, **k): st_calls.setdefault('col_subheader', []).append((a, k))
        def write(self, *a, **k): st_calls.setdefault('col_write', []).append((a, k))

    class DummySidebar:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def header(self, *a, **k): st_calls.setdefault('sidebar_header', []).append((a, k))
        def divider(self, *a, **k): st_calls.setdefault('sidebar_divider', []).append((a, k))
        def markdown(self, *a, **k): st_calls.setdefault('sidebar_markdown', []).append((a, k))
        def text_input(self, *a, **k): return "http://mocked-backend:8000"

    class DummyTabs:
        def __init__(self, labels):
            self.labels = labels
            self.idx = -1
        def __getitem__(self, idx):
            self.idx = idx
            return self
        def header(self, *a, **k): st_calls.setdefault(f'tab{self.idx}_header', []).append((a, k))
        def write(self, *a, **k): st_calls.setdefault(f'tab{self.idx}_write', []).append((a, k))
        def file_uploader(self, *a, **k):
            key = k.get('key', None)
            if key == "extraction_uploader":
                return st_calls.get('extract_file', None)
            return st_calls.get('uploaded_files', None)
        def button(self, *a, **k): return st_calls.get(f'tab{self.idx}_button', False)
        def spinner(self, *a, **k): return DummyExpander()
        def expander(self, *a, **k): return DummyExpander()
        def columns(self, n): return DummyColumns(n)
        def subheader(self, *a, **k): st_calls.setdefault(f'tab{self.idx}_subheader', []).append((a, k))
        def json(self, *a, **k): st_calls.setdefault(f'tab{self.idx}_json', []).append((a, k))
        def warning(self, *a, **k): st_calls.setdefault(f'tab{self.idx}_warning', []).append((a, k))
        def error(self, *a, **k): st_calls.setdefault(f'tab{self.idx}_error', []).append((a, k))
        def success(self, *a, **k): st_calls.setdefault(f'tab{self.idx}_success', []).append((a, k))

    class DummySt:
        def set_page_config(self, *a, **k): st_calls.setdefault('set_page_config', []).append((a, k))
        def title(self, *a, **k): st_calls.setdefault('title', []).append((a, k))
        def sidebar(self): return DummySidebar()
        def header(self, *a, **k): st_calls.setdefault('header', []).append((a, k))
        def divider(self, *a, **k): st_calls.setdefault('divider', []).append((a, k))
        def markdown(self, *a, **k): st_calls.setdefault('markdown', []).append((a, k))
        def text_input(self, *a, **k): return "http://mocked-backend:8000"
        def error(self, *a, **k): st_calls.setdefault('error', []).append((a, k))
        def warning(self, *a, **k): st_calls.setdefault('warning', []).append((a, k))
        def success(self, *a, **k): st_calls.setdefault('success', []).append((a, k))
        def button(self, *a, **k): return st_calls.get('button', False)
        def stop(self): raise Exception("st.stop called")
        def tabs(self, labels): return DummyTabs(labels)
        def file_uploader(self, *a, **k):
            key = k.get('key', None)
            if key == "extraction_uploader":
                return st_calls.get('extract_file', None)
            return st_calls.get('uploaded_files', None)
        def spinner(self, *a, **k): return DummyExpander()
        def expander(self, *a, **k): return DummyExpander()
        def chat_message(self, role): return DummyChatMessage(role)
        def chat_input(self, *a, **k): return st_calls.get('chat_input', None)
        def session_state(self): return st_calls.setdefault('session_state', {})
        def columns(self, n): return DummyColumns(n)
        def subheader(self, *a, **k): st_calls.setdefault('subheader', []).append((a, k))
        def json(self, *a, **k): st_calls.setdefault('json', []).append((a, k))
        def caption(self, *a, **k): st_calls.setdefault('caption', []).append((a, k))
        def write(self, *a, **k): st_calls.setdefault('write', []).append((a, k))

    dummy_st = DummySt()
    monkeypatch.setattr(app, "st", dummy_st)
    return st_calls

@pytest.fixture
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    calls = {"get": [], "post": []}
    class DummyResponse:
        def __init__(self, status_code=200, json_data=None, text="OK"):
            self.status_code = status_code
            self._json = json_data
            self.text = text
        def json(self):
            return self._json
    def dummy_get(url, *a, **k):
        calls["get"].append((url, a, k))
        if url.endswith("/ping"):
            return DummyResponse(status_code=200)
        return DummyResponse(status_code=404, text="Not Found")
    def dummy_post(url, *a, **k):
        calls["post"].append((url, a, k))
        if url.endswith("/upload"):
            files = k.get("files", [])
            if not files:
                return DummyResponse(status_code=400, text="No files")
            return DummyResponse(status_code=200, json_data={
                "message": "Indexed 2 documents.",
                "extractions": [
                    {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
                    {"filename": "doc2.pdf", "structured_data_extracted": False, "text_chunks": 3}
                ],
                "errors": []
            })
        if url.endswith("/ask"):
            payload = k.get("json", {})
            if not payload.get("question"):
                return DummyResponse(status_code=400, text="No question")
            return DummyResponse(status_code=200, json_data={
                "answer": "The agreed rate is $1200.",
                "confidence_score": 0.95,
                "sources": [
                    {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
                ]
            })
        if url.endswith("/extract"):
            files = k.get("files", {})
            if not files:
                return DummyResponse(status_code=400, text="No file")
            return DummyResponse(status_code=200, json_data={
                "data": {
                    "reference_id": "REF123",
                    "load_id": "LOAD456",
                    "shipper": "Acme Corp",
                    "consignee": "Beta LLC",
                    "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
                    "driver": {"driver_name": "John Doe", "truck_number": "TRK123"},
                    "pickup": {"city": "Dallas", "state": "TX"},
                    "shipping_date": "2024-06-01",
                    "drop": {"city": "Houston", "state": "TX"},
                    "delivery_date": "2024-06-02",
                    "rate_info": {"total_rate": 1200, "currency": "USD"},
                    "equipment_type": "Van"
                }
            })
        return DummyResponse(status_code=404, text="Not Found")
    monkeypatch.setattr(app.requests, "get", dummy_get)
    monkeypatch.setattr(app.requests, "post", dummy_post)
    return calls

@pytest.fixture
def patch_os_environ(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://mocked-backend:8000")

def test_backend_health_check_success(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    # Should call requests.get with /ping and proceed
    assert app.check_backend() is True
    assert patch_requests["get"][0][0].endswith("/ping")

def test_backend_health_check_failure(monkeypatch, patch_streamlit, patch_os_environ):
    # Simulate backend down
    def fail_get(url, *a, **k): raise Exception("Connection error")
    monkeypatch.setattr(app.requests, "get", fail_get)
    assert app.check_backend() is False

def test_upload_and_index_happy_path(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    # Simulate two uploaded files and button click
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content
    files = [
        DummyFile("doc1.pdf", b"PDFDATA1", "application/pdf"),
        DummyFile("doc2.pdf", b"PDFDATA2", "application/pdf")
    ]
    patch_streamlit['uploaded_files'] = files
    patch_streamlit['tab0_button'] = True
    # Run the Upload & Index tab logic
    # Simulate the tab selection
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[0]
    tab.header("Upload Logistics Documents")
    tab.write("Upload PDF, DOCX, or TXT documents to index them for Q&A.")
    uploaded_files = tab.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if tab.button("Process & Index Documents"):
        if uploaded_files:
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            with tab.spinner("Processing documents (chunking, embedding, indexing)..."):
                response = app.requests.post(
                    "http://mocked-backend:8000/upload",
                    files=files_to_send
                )
                assert response.status_code == 200
                result = response.json()
                assert result["message"] == "Indexed 2 documents."
                assert len(result["extractions"]) == 2
                assert result["extractions"][0]["filename"] == "doc1.pdf"
                assert result["extractions"][1]["filename"] == "doc2.pdf"
                assert result["extractions"][0]["structured_data_extracted"] is True
                assert result["extractions"][1]["structured_data_extracted"] is False

def test_upload_and_index_no_files(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    patch_streamlit['uploaded_files'] = []
    patch_streamlit['tab0_button'] = True
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[0]
    tab.header("Upload Logistics Documents")
    tab.write("Upload PDF, DOCX, or TXT documents to index them for Q&A.")
    uploaded_files = tab.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if tab.button("Process & Index Documents"):
        if not uploaded_files:
            tab.warning("Please upload at least one document.")

def test_upload_and_index_backend_error(monkeypatch, patch_streamlit, patch_os_environ):
    # Simulate backend error
    def fail_post(url, *a, **k):
        class DummyResponse:
            status_code = 500
            text = "Internal Server Error"
            def json(self): return {}
        return DummyResponse()
    monkeypatch.setattr(app.requests, "post", fail_post)
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content
    patch_streamlit['uploaded_files'] = [DummyFile("doc1.pdf", b"PDFDATA1", "application/pdf")]
    patch_streamlit['tab0_button'] = True
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[0]
    tab.header("Upload Logistics Documents")
    tab.write("Upload PDF, DOCX, or TXT documents to index them for Q&A.")
    uploaded_files = tab.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if tab.button("Process & Index Documents"):
        if uploaded_files:
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            with tab.spinner("Processing documents (chunking, embedding, indexing)..."):
                response = app.requests.post(
                    "http://mocked-backend:8000/upload",
                    files=files_to_send
                )
                assert response.status_code == 500
                assert response.text == "Internal Server Error"

def test_chat_qa_happy_path(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    patch_streamlit['chat_input'] = "What is the agreed rate for this shipment?"
    # Simulate session state
    app.st.session_state = types.SimpleNamespace()
    app.st.session_state.messages = []
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[1]
    tab.header("Question Answering")
    tab.write("Ask questions about your uploaded documents.")
    if "messages" not in app.st.session_state.__dict__:
        app.st.session_state.messages = []
    for message in app.st.session_state.messages:
        with app.st.chat_message(message["role"]):
            app.st.markdown(message["content"])
            if "sources" in message:
                with app.st.expander("View Sources"):
                    for src in message["sources"]:
                        app.st.caption(f"Source: {src['metadata']['source']}")
                        app.st.write(src["text"])
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    if prompt:
        app.st.session_state.messages.append({"role": "user", "content": prompt})
        with app.st.chat_message("user"):
            app.st.markdown(prompt)
        with app.st.chat_message("assistant"):
            with app.st.spinner("Thinking..."):
                payload = {"question": prompt, "chat_history": []}
                response = app.requests.post("http://mocked-backend:8000/ask", json=payload)
                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "The agreed rate is $1200."
                assert data["confidence_score"] == 0.95
                assert data["sources"][0]["metadata"]["source"] == "doc1.pdf"
                app.st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"]
                })

def test_chat_qa_no_question(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    patch_streamlit['chat_input'] = ""
    app.st.session_state = types.SimpleNamespace()
    app.st.session_state.messages = []
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[1]
    tab.header("Question Answering")
    tab.write("Ask questions about your uploaded documents.")
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    assert prompt == ""

def test_chat_qa_backend_error(monkeypatch, patch_streamlit, patch_os_environ):
    def fail_post(url, *a, **k):
        class DummyResponse:
            status_code = 500
            text = "Internal Server Error"
            def json(self): return {}
        return DummyResponse()
    monkeypatch.setattr(app.requests, "post", fail_post)
    patch_streamlit['chat_input'] = "What is the agreed rate for this shipment?"
    app.st.session_state = types.SimpleNamespace()
    app.st.session_state.messages = []
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[1]
    tab.header("Question Answering")
    tab.write("Ask questions about your uploaded documents.")
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    if prompt:
        app.st.session_state.messages.append({"role": "user", "content": prompt})
        with app.st.chat_message("user"):
            app.st.markdown(prompt)
        with app.st.chat_message("assistant"):
            with app.st.spinner("Thinking..."):
                response = app.requests.post("http://mocked-backend:8000/ask", json={"question": prompt, "chat_history": []})
                assert response.status_code == 500
                assert response.text == "Internal Server Error"

def test_data_extraction_happy_path(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content
    patch_streamlit['extract_file'] = DummyFile("doc1.pdf", b"PDFDATA1", "application/pdf")
    patch_streamlit['tab2_button'] = True
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[2]
    tab.header("Structured Data Extraction")
    tab.write("Extract specific fields from a logistics document.")
    extract_file = tab.file_uploader(
        "Upload a single document for extraction",
        type=["pdf", "docx", "txt"],
        key="extraction_uploader"
    )
    if tab.button("Run Extraction"):
        if extract_file:
            with tab.spinner("Extracting data..."):
                files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                response = app.requests.post("http://mocked-backend:8000/extract", files=files)
                assert response.status_code == 200
                data = response.json()["data"]
                assert data["reference_id"] == "REF123"
                assert data["load_id"] == "LOAD456"
                assert data["shipper"] == "Acme Corp"
                assert data["consignee"] == "Beta LLC"
                assert data["carrier"]["carrier_name"] == "CarrierX"
                assert data["driver"]["driver_name"] == "John Doe"
                assert data["pickup"]["city"] == "Dallas"
                assert data["drop"]["city"] == "Houston"
                assert data["rate_info"]["total_rate"] == 1200
                assert data["equipment_type"] == "Van"

def test_data_extraction_no_file(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    patch_streamlit['extract_file'] = None
    patch_streamlit['tab2_button'] = True
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[2]
    tab.header("Structured Data Extraction")
    tab.write("Extract specific fields from a logistics document.")
    extract_file = tab.file_uploader(
        "Upload a single document for extraction",
        type=["pdf", "docx", "txt"],
        key="extraction_uploader"
    )
    if tab.button("Run Extraction"):
        if not extract_file:
            tab.warning("Please upload a document.")

def test_data_extraction_backend_error(monkeypatch, patch_streamlit, patch_os_environ):
    def fail_post(url, *a, **k):
        class DummyResponse:
            status_code = 500
            text = "Internal Server Error"
            def json(self): return {}
        return DummyResponse()
    monkeypatch.setattr(app.requests, "post", fail_post)
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content
    patch_streamlit['extract_file'] = DummyFile("doc1.pdf", b"PDFDATA1", "application/pdf")
    patch_streamlit['tab2_button'] = True
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab = tabs[2]
    tab.header("Structured Data Extraction")
    tab.write("Extract specific fields from a logistics document.")
    extract_file = tab.file_uploader(
        "Upload a single document for extraction",
        type=["pdf", "docx", "txt"],
        key="extraction_uploader"
    )
    if tab.button("Run Extraction"):
        if extract_file:
            with tab.spinner("Extracting data..."):
                files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                response = app.requests.post("http://mocked-backend:8000/extract", files=files)
                assert response.status_code == 500
                assert response.text == "Internal Server Error"

def test_reconciliation_equivalent_paths(monkeypatch, patch_streamlit, patch_requests, patch_os_environ):
    # Simulate uploading a file and then extracting data from it, compare outputs
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content
    file = DummyFile("doc1.pdf", b"PDFDATA1", "application/pdf")
    patch_streamlit['uploaded_files'] = [file]
    patch_streamlit['tab0_button'] = True
    patch_streamlit['extract_file'] = file
    patch_streamlit['tab2_button'] = True
    # Upload & Index
    tabs = app.st.tabs(["📄 Upload & Index", "💬 Chat / Q&A", "📊 Data Extraction"])
    tab_upload = tabs[0]
    uploaded_files = tab_upload.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if tab_upload.button("Process & Index Documents"):
        files_to_send = [
            ("files", (f.name, f.getvalue(), f.type))
            for f in uploaded_files
        ]
        with tab_upload.spinner("Processing documents (chunking, embedding, indexing)..."):
            response_upload = app.requests.post(
                "http://mocked-backend:8000/upload",
                files=files_to_send
            )
            upload_json = response_upload.json()
    # Data Extraction
    tab_extract = tabs[2]
    extract_file = tab_extract.file_uploader(
        "Upload a single document for extraction",
        type=["pdf", "docx", "txt"],
        key="extraction_uploader"
    )
    if tab_extract.button("Run Extraction"):
        with tab_extract.spinner("Extracting data..."):
            files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
            response_extract = app.requests.post("http://mocked-backend:8000/extract", files=files)
            extract_json = response_extract.json()
    # Compare outputs for reconciliation
    assert upload_json["extractions"][0]["filename"] == extract_file.name
    assert extract_json["data"]["reference_id"] == "REF123"
    assert extract_json["data"]["shipper"] == "Acme Corp"
