# source_hash: 9ee607a3d8da4254
# import_target: frontend.app
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vs" not in st.session_state:
    st.session_state["vs"] = None

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
import builtins
import types
import io

import frontend.app

import streamlit as st

from unittest import mock

@pytest.fixture(autouse=True)
def reset_streamlit_session_state(monkeypatch):
    # Bootstrap session_state for messages and vs
    if hasattr(st, "session_state"):
        st.session_state.clear()
    else:
        st.session_state = {}
    st.session_state["messages"] = []
    st.session_state["vs"] = {}
    yield
    st.session_state.clear()

@pytest.fixture
def mock_streamlit(monkeypatch):
    # Patch all streamlit UI functions to no-ops or record calls
    monkeypatch.setattr(st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(st, "title", lambda *a, **k: None)
    monkeypatch.setattr(st, "header", lambda *a, **k: None)
    monkeypatch.setattr(st, "write", lambda *a, **k: None)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "divider", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(st, "spinner", lambda *a, **k: (v for v in [None]))
    monkeypatch.setattr(st, "expander", lambda *a, **k: mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))
    monkeypatch.setattr(st, "columns", lambda n: [mock.MagicMock(), mock.MagicMock()])
    monkeypatch.setattr(st, "json", lambda *a, **k: None)
    monkeypatch.setattr(st, "tabs", lambda labels: [mock.MagicMock() for _ in labels])
    monkeypatch.setattr(st, "chat_message", lambda *a, **k: mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))
    monkeypatch.setattr(st, "chat_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: None)
    monkeypatch.setattr(st, "button", lambda *a, **k: False)
    monkeypatch.setattr(st, "text_input", lambda *a, **k: "http://localhost:8000")
    yield

@pytest.fixture
def mock_requests(monkeypatch):
    # Patch requests.get and requests.post
    with mock.patch("frontend.app.requests.get") as mock_get, \
         mock.patch("frontend.app.requests.post") as mock_post:
        yield mock_get, mock_post

@pytest.fixture
def mock_os_environ(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend:9999")
    yield

def make_file_mock(name, content, filetype):
    file_mock = mock.MagicMock()
    file_mock.name = name
    file_mock.type = filetype
    file_mock.getvalue.return_value = content
    return file_mock

def test_backend_health_happy_path(mock_streamlit, mock_requests):
    mock_get, _ = mock_requests
    mock_get.return_value.status_code = 200
    # Should not raise SystemExit (st.stop)
    try:
        frontend.app.check_backend()
    except SystemExit:
        pytest.fail("Should not stop app when backend is healthy")
    assert frontend.app.check_backend() is True

def test_backend_health_unreachable(mock_streamlit, mock_requests):
    mock_get, _ = mock_requests
    mock_get.side_effect = Exception("Connection error")
    assert frontend.app.check_backend() is False

def test_backend_health_non_200(mock_streamlit, mock_requests):
    mock_get, _ = mock_requests
    mock_get.return_value.status_code = 503
    assert frontend.app.check_backend() is False

def test_upload_and_index_documents_happy_path(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    # Simulate uploaded files
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    file2 = make_file_mock("doc2.txt", b"TXTDATA", "text/plain")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [file1, file2])
    monkeypatch.setattr(st, "button", lambda label: label == "Process & Index Documents")
    # Simulate backend response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "message": "Indexed 2 documents.",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
            {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    }
    # Patch requests.get for health check
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    # Run app (should not raise)
    try:
        import importlib
        importlib.reload(frontend.app)
    except SystemExit:
        pytest.fail("Should not stop app on successful upload and index")

def test_upload_and_index_documents_no_files(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [])
    monkeypatch.setattr(st, "button", lambda label: label == "Process & Index Documents")
    # Patch requests.get for health check
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    # Should not call requests.post
    import importlib
    importlib.reload(frontend.app)
    assert not mock_post.called

def test_upload_and_index_documents_backend_error(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [file1])
    monkeypatch.setattr(st, "button", lambda label: label == "Process & Index Documents")
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    import importlib
    importlib.reload(frontend.app)
    assert mock_post.called

def test_upload_and_index_documents_request_exception(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [file1])
    monkeypatch.setattr(st, "button", lambda label: label == "Process & Index Documents")
    mock_post.side_effect = Exception("Network error")
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    import importlib
    importlib.reload(frontend.app)
    assert mock_post.called

def test_chat_qa_happy_path(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    # Simulate chat input
    monkeypatch.setattr(st, "chat_input", lambda *a, **k: "What is the agreed rate?")
    # Patch chat_message context manager
    monkeypatch.setattr(st, "chat_message", lambda *a, **k: mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))
    # Patch requests.get for health check
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    # Simulate backend response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
        ]
    }
    # Patch st.session_state
    st.session_state["messages"] = []
    import importlib
    importlib.reload(frontend.app)
    assert st.session_state["messages"][-1]["role"] == "assistant"
    assert "agreed rate" in st.session_state["messages"][-1]["content"]

def test_chat_qa_backend_error(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    monkeypatch.setattr(st, "chat_input", lambda *a, **k: "What is the agreed rate?")
    monkeypatch.setattr(st, "chat_message", lambda *a, **k: mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"
    st.session_state["messages"] = []
    import importlib
    importlib.reload(frontend.app)
    assert st.session_state["messages"][-1]["role"] == "user"

def test_chat_qa_request_exception(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    monkeypatch.setattr(st, "chat_input", lambda *a, **k: "What is the agreed rate?")
    monkeypatch.setattr(st, "chat_message", lambda *a, **k: mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    mock_post.side_effect = Exception("Network error")
    st.session_state["messages"] = []
    import importlib
    importlib.reload(frontend.app)
    assert st.session_state["messages"][-1]["role"] == "user"

def test_data_extraction_happy_path(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: file1)
    monkeypatch.setattr(st, "button", lambda label: label == "Run Extraction")
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Shipper Inc.",
            "consignee": "Consignee LLC",
            "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
            "driver": {"driver_name": "John Doe", "truck_number": "TRK123"},
            "pickup": {"city": "Dallas", "state": "TX"},
            "shipping_date": "2023-01-01",
            "drop": {"city": "Houston", "state": "TX"},
            "delivery_date": "2023-01-02",
            "rate_info": {"total_rate": "1200", "currency": "USD"},
            "equipment_type": "Van"
        }
    }
    import importlib
    importlib.reload(frontend.app)
    assert mock_post.called

def test_data_extraction_no_file(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: None)
    monkeypatch.setattr(st, "button", lambda label: label == "Run Extraction")
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    import importlib
    importlib.reload(frontend.app)
    assert not mock_post.called

def test_data_extraction_backend_error(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: file1)
    monkeypatch.setattr(st, "button", lambda label: label == "Run Extraction")
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"
    import importlib
    importlib.reload(frontend.app)
    assert mock_post.called

def test_data_extraction_request_exception(monkeypatch, mock_streamlit, mock_requests):
    _, mock_post = mock_requests
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: file1)
    monkeypatch.setattr(st, "button", lambda label: label == "Run Extraction")
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    mock_post.side_effect = Exception("Network error")
    import importlib
    importlib.reload(frontend.app)
    assert mock_post.called

def test_backend_unreachable_stops_app(monkeypatch, mock_streamlit, mock_requests):
    mock_get, _ = mock_requests
    mock_get.side_effect = Exception("Connection error")
    # Should raise SystemExit due to st.stop
    with pytest.raises(SystemExit):
        import importlib
        importlib.reload(frontend.app)

def test_backend_unreachable_retry(monkeypatch, mock_streamlit, mock_requests):
    mock_get, _ = mock_requests
    mock_get.side_effect = Exception("Connection error")
    # Simulate Retry Connection button pressed
    monkeypatch.setattr(st, "button", lambda label: label == "Retry Connection")
    monkeypatch.setattr(st, "rerun", lambda: (_ for _ in ()).throw(SystemExit))
    with pytest.raises(SystemExit):
        import importlib
        importlib.reload(frontend.app)

def test_sidebar_backend_url_env(monkeypatch, mock_streamlit, mock_requests, mock_os_environ):
    # Should use BACKEND_URL from env
    monkeypatch.setattr(frontend.app.requests, "get", lambda url: mock.Mock(status_code=200))
    import importlib
    importlib.reload(frontend.app)
    # The text_input should be called with value="http://test-backend:9999"
    assert st.text_input("Backend API URL", value="http://test-backend:9999", help=mock.ANY) == "http://localhost:8000"
