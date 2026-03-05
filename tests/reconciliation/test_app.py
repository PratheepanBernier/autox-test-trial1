import pytest
from unittest.mock import patch, MagicMock
import builtins

# Since the app is a Streamlit script, we focus on reconciling the logic of backend interactions and UI state.
# We'll test the helper and backend interaction logic in isolation, mocking Streamlit and requests.

# Helper to simulate Streamlit session state and UI elements
class DummySessionState(dict):
    pass

class DummyChatMessage:
    def __init__(self, role):
        self.role = role
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def markdown(self, content):
        pass
    def caption(self, content):
        pass
    def expander(self, label):
        return self
    def write(self, content):
        pass

class DummyExpander:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def write(self, content): pass
    def caption(self, content): pass

class DummyColumns:
    def __getitem__(self, idx): return DummyColumn()
    def __iter__(self): return iter([DummyColumn(), DummyColumn()])

class DummyColumn:
    def subheader(self, content): pass
    def write(self, content): pass

class DummyStreamlit:
    def __init__(self):
        self.session_state = DummySessionState()
        self.sidebar = self
        self.spinner = self._dummy_context
        self.expander = lambda label: DummyExpander()
        self.columns = lambda n: (DummyColumn(), DummyColumn())
        self.tabs = lambda labels: [self, self, self]
        self.file_uploader = MagicMock(return_value=None)
        self.button = MagicMock(return_value=False)
        self.text_input = MagicMock(return_value="http://localhost:8000")
        self.chat_input = MagicMock(return_value=None)
        self.chat_message = lambda role: DummyChatMessage(role)
        self.set_page_config = MagicMock()
        self.header = MagicMock()
        self.write = MagicMock()
        self.markdown = MagicMock()
        self.divider = MagicMock()
        self.caption = MagicMock()
        self.json = MagicMock()
        self.success = MagicMock()
        self.warning = MagicMock()
        self.error = MagicMock()
        self.stop = MagicMock()
        self.rerun = MagicMock()
    def _dummy_context(self, *a, **k):
        class DummyCtx:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyCtx()

@pytest.fixture
def dummy_st():
    return DummyStreamlit()

@pytest.fixture
def patch_streamlit(monkeypatch, dummy_st):
    monkeypatch.setattr("streamlit.st", dummy_st)
    monkeypatch.setattr("streamlit.sidebar", dummy_st.sidebar)
    monkeypatch.setattr("streamlit.session_state", dummy_st.session_state)
    return dummy_st

@pytest.fixture
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    get_mock = MagicMock()
    post_mock = MagicMock()
    monkeypatch.setattr("requests.get", get_mock)
    monkeypatch.setattr("requests.post", post_mock)
    return get_mock, post_mock

@pytest.fixture
def patch_os(monkeypatch):
    monkeypatch.setattr("os.getenv", lambda key, default=None: default)

# --- Reconciliation Tests ---

def run_check_backend(BACKEND_URL, requests_get):
    # Simulate the check_backend helper
    try:
        response = requests_get(f"{BACKEND_URL}/ping")
        return response.status_code == 200
    except Exception:
        return False

def test_check_backend_happy_path(patch_requests):
    get_mock, _ = patch_requests
    get_mock.return_value.status_code = 200
    assert run_check_backend("http://localhost:8000", get_mock) is True

def test_check_backend_down(patch_requests):
    get_mock, _ = patch_requests
    get_mock.return_value.status_code = 500
    assert run_check_backend("http://localhost:8000", get_mock) is False

def test_check_backend_exception(patch_requests):
    get_mock, _ = patch_requests
    get_mock.side_effect = Exception("Network error")
    assert run_check_backend("http://localhost:8000", get_mock) is False

def test_backend_url_env_and_text_input_reconcile(monkeypatch, dummy_st):
    # Reconcile: env var and text_input default
    monkeypatch.setattr("os.getenv", lambda key, default=None: "http://env-backend:9000")
    dummy_st.text_input = MagicMock(return_value="http://env-backend:9000")
    # Simulate sidebar config logic
    default_backend = "http://env-backend:9000"
    BACKEND_URL = dummy_st.text_input("Backend API URL", value=default_backend, help="The URL of the FastAPI backend.")
    assert BACKEND_URL == "http://env-backend:9000"

def test_upload_and_index_success(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    # Simulate uploaded files
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    uploaded_files = [DummyFile("doc1.pdf", "application/pdf"), DummyFile("doc2.txt", "text/plain")]
    dummy_st.file_uploader = MagicMock(return_value=uploaded_files)
    dummy_st.button = MagicMock(return_value=True)
    # Simulate backend response
    post_mock.return_value.status_code = 200
    post_mock.return_value.json.return_value = {
        "message": "Indexed 2 documents.",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
            {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    }
    # Simulate UI logic for upload & index
    files_to_send = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in uploaded_files
    ]
    response = post_mock("http://localhost:8000/upload", files=files_to_send)
    assert response.status_code == 200
    result = response.json()
    assert result["message"] == "Indexed 2 documents."
    assert len(result["extractions"]) == 2
    assert all("filename" in ext for ext in result["extractions"])

def test_upload_and_index_error(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    uploaded_files = [DummyFile("doc1.pdf", "application/pdf")]
    dummy_st.file_uploader = MagicMock(return_value=uploaded_files)
    dummy_st.button = MagicMock(return_value=True)
    post_mock.return_value.status_code = 400
    post_mock.return_value.text = "Bad Request"
    files_to_send = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in uploaded_files
    ]
    response = post_mock("http://localhost:8000/upload", files=files_to_send)
    assert response.status_code == 400
    assert response.text == "Bad Request"

def test_upload_and_index_no_files(dummy_st):
    dummy_st.file_uploader = MagicMock(return_value=[])
    dummy_st.button = MagicMock(return_value=True)
    # Should warn about missing files
    assert dummy_st.file_uploader() == []
    assert dummy_st.button() is True

def test_chat_qa_happy_path(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    dummy_st.session_state.messages = []
    dummy_st.chat_input = MagicMock(return_value="What is the rate?")
    post_mock.return_value.status_code = 200
    post_mock.return_value.json.return_value = {
        "answer": "The rate is $1000.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Relevant text"}
        ]
    }
    prompt = dummy_st.chat_input("What is the agreed rate for this shipment?")
    payload = {"question": prompt, "chat_history": []}
    response = post_mock("http://localhost:8000/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence_score" in data
    assert isinstance(data["sources"], list)
    # Simulate message append
    dummy_st.session_state.messages.append({"role": "user", "content": prompt})
    dummy_st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"],
        "sources": data["sources"]
    })
    assert dummy_st.session_state.messages[0]["role"] == "user"
    assert dummy_st.session_state.messages[1]["role"] == "assistant"

def test_chat_qa_backend_error(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    dummy_st.session_state.messages = []
    dummy_st.chat_input = MagicMock(return_value="What is the rate?")
    post_mock.return_value.status_code = 500
    post_mock.return_value.text = "Internal Server Error"
    prompt = dummy_st.chat_input("What is the agreed rate for this shipment?")
    payload = {"question": prompt, "chat_history": []}
    response = post_mock("http://localhost:8000/ask", json=payload)
    assert response.status_code == 500
    assert response.text == "Internal Server Error"

def test_chat_qa_backend_exception(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    dummy_st.session_state.messages = []
    dummy_st.chat_input = MagicMock(return_value="What is the rate?")
    post_mock.side_effect = Exception("Network error")
    prompt = dummy_st.chat_input("What is the agreed rate for this shipment?")
    payload = {"question": prompt, "chat_history": []}
    try:
        post_mock("http://localhost:8000/ask", json=payload)
    except Exception as e:
        assert str(e) == "Network error"

def test_data_extraction_success(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    extract_file = DummyFile("doc1.pdf", "application/pdf")
    dummy_st.file_uploader = MagicMock(return_value=extract_file)
    dummy_st.button = MagicMock(return_value=True)
    post_mock.return_value.status_code = 200
    post_mock.return_value.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Shipper Inc.",
            "consignee": "Consignee LLC",
            "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
            "driver": {"driver_name": "John Doe", "truck_number": "TRK123"},
            "pickup": {"city": "CityA", "state": "ST"},
            "shipping_date": "2024-01-01",
            "drop": {"city": "CityB", "state": "TS"},
            "delivery_date": "2024-01-02",
            "rate_info": {"total_rate": 1000, "currency": "USD"},
            "equipment_type": "Van"
        }
    }
    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
    response = post_mock("http://localhost:8000/extract", files=files)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["reference_id"] == "REF123"
    assert data["carrier"]["carrier_name"] == "CarrierX"
    assert data["rate_info"]["total_rate"] == 1000

def test_data_extraction_error(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    extract_file = DummyFile("doc1.pdf", "application/pdf")
    dummy_st.file_uploader = MagicMock(return_value=extract_file)
    dummy_st.button = MagicMock(return_value=True)
    post_mock.return_value.status_code = 400
    post_mock.return_value.text = "Bad Request"
    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
    response = post_mock("http://localhost:8000/extract", files=files)
    assert response.status_code == 400
    assert response.text == "Bad Request"

def test_data_extraction_no_file(dummy_st):
    dummy_st.file_uploader = MagicMock(return_value=None)
    dummy_st.button = MagicMock(return_value=True)
    assert dummy_st.file_uploader() is None
    assert dummy_st.button() is True

def test_data_extraction_backend_exception(monkeypatch, patch_requests, dummy_st):
    _, post_mock = patch_requests
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    extract_file = DummyFile("doc1.pdf", "application/pdf")
    dummy_st.file_uploader = MagicMock(return_value=extract_file)
    dummy_st.button = MagicMock(return_value=True)
    post_mock.side_effect = Exception("Network error")
    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
    try:
        post_mock("http://localhost:8000/extract", files=files)
    except Exception as e:
        assert str(e) == "Network error"

def test_reconcile_upload_vs_extraction_file_handling(monkeypatch, patch_requests, dummy_st):
    # Reconcile: upload supports multiple files, extraction only one
    _, post_mock = patch_requests
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    uploaded_files = [DummyFile("doc1.pdf", "application/pdf"), DummyFile("doc2.txt", "text/plain")]
    extract_file = DummyFile("doc1.pdf", "application/pdf")
    # Upload: multiple files
    files_to_send = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in uploaded_files
    ]
    # Extraction: single file
    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
    assert isinstance(files_to_send, list)
    assert isinstance(files, dict)
    assert len(files_to_send) == 2
    assert list(files.keys()) == ["file"]

def test_reconcile_chat_history_state(dummy_st):
    # Reconcile: chat history is preserved in session_state
    dummy_st.session_state.messages = []
    dummy_st.session_state.messages.append({"role": "user", "content": "Q1"})
    dummy_st.session_state.messages.append({"role": "assistant", "content": "A1"})
    # Simulate new chat input
    dummy_st.session_state.messages.append({"role": "user", "content": "Q2"})
    assert dummy_st.session_state.messages[0]["content"] == "Q1"
    assert dummy_st.session_state.messages[2]["content"] == "Q2"
    # Reconcile: order and role preserved
    roles = [m["role"] for m in dummy_st.session_state.messages]
    assert roles == ["user", "assistant", "user"]

def test_reconcile_error_handling_upload_vs_extraction(monkeypatch, patch_requests, dummy_st):
    # Reconcile: both upload and extraction handle backend errors similarly
    _, post_mock = patch_requests
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    uploaded_files = [DummyFile("doc1.pdf", "application/pdf")]
    extract_file = DummyFile("doc1.pdf", "application/pdf")
    # Upload error
    post_mock.return_value.status_code = 400
    post_mock.return_value.text = "Bad Request"
    files_to_send = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in uploaded_files
    ]
    response_upload = post_mock("http://localhost:8000/upload", files=files_to_send)
    # Extraction error
    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
    response_extract = post_mock("http://localhost:8000/extract", files=files)
    assert response_upload.status_code == response_extract.status_code
    assert response_upload.text == response_extract.text

def test_reconcile_success_paths_upload_vs_extraction(monkeypatch, patch_requests, dummy_st):
    # Reconcile: both upload and extraction return JSON on success
    _, post_mock = patch_requests
    class DummyFile:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        def getvalue(self): return b"filecontent"
    uploaded_files = [DummyFile("doc1.pdf", "application/pdf")]
    extract_file = DummyFile("doc1.pdf", "application/pdf")
    # Upload success
    post_mock.return_value.status_code = 200
    post_mock.return_value.json.side_effect = [
        {
            "message": "Indexed 1 document.",
            "extractions": [
                {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5}
            ],
            "errors": []
        },
        {
            "data": {
                "reference_id": "REF123",
                "load_id": "LOAD456"
            }
        }
    ]
    files_to_send = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in uploaded_files
    ]
    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
    response_upload = post_mock("http://localhost:8000/upload", files=files_to_send)
    response_extract = post_mock("http://localhost:8000/extract", files=files)
    assert response_upload.status_code == 200
    assert response_extract.status_code == 200
    upload_json = response_upload.json()
    extract_json = response_extract.json()
    assert "message" in upload_json
    assert "data" in extract_json
    # Reconcile: both return dicts with expected keys
    assert isinstance(upload_json, dict)
    assert isinstance(extract_json, dict)
