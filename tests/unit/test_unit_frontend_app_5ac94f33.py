# source_hash: 9ee607a3d8da4254
# import_target: frontend.app
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
import builtins
import types
from unittest.mock import patch, MagicMock, call

import frontend.app as app

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    st_mock = MagicMock()
    monkeypatch.setattr(app, "st", st_mock)
    return st_mock

@pytest.fixture(autouse=True)
def patch_requests(monkeypatch):
    requests_mock = MagicMock()
    monkeypatch.setattr(app, "requests", requests_mock)
    return requests_mock

@pytest.fixture(autouse=True)
def patch_os(monkeypatch):
    os_mock = MagicMock()
    monkeypatch.setattr(app, "os", os_mock)
    return os_mock

def test_check_backend_success(patch_requests):
    patch_requests.get.return_value.status_code = 200
    app.BACKEND_URL = "http://test"
    assert app.check_backend() is True
    patch_requests.get.assert_called_once_with("http://test/ping")

def test_check_backend_failure_status(patch_requests):
    patch_requests.get.return_value.status_code = 500
    app.BACKEND_URL = "http://test"
    assert app.check_backend() is False

def test_check_backend_exception(patch_requests):
    patch_requests.get.side_effect = Exception("fail")
    app.BACKEND_URL = "http://test"
    assert app.check_backend() is False

def test_backend_url_from_env(monkeypatch, patch_streamlit, patch_os):
    patch_os.getenv.return_value = "http://env-backend"
    patch_streamlit.text_input.return_value = "http://env-backend"
    # Simulate sidebar context manager
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    # Re-import to trigger sidebar logic
    import importlib
    importlib.reload(app)
    patch_os.getenv.assert_called_with("BACKEND_URL", "http://localhost:8000")
    patch_streamlit.text_input.assert_called_with(
        "Backend API URL",
        value="http://env-backend",
        help="The URL of the FastAPI backend."
    )

def test_backend_down_shows_error_and_retry(monkeypatch, patch_streamlit, patch_requests):
    # Simulate backend down
    monkeypatch.setattr(app, "check_backend", lambda: False)
    patch_streamlit.button.return_value = False
    patch_streamlit.stop.side_effect = Exception("stop called")
    # Simulate sidebar context manager
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    # Re-import to trigger logic
    import importlib
    try:
        importlib.reload(app)
    except Exception as e:
        assert "stop called" in str(e)
    patch_streamlit.error.assert_any_call("⚠️ Cannot connect to backend at {}. Please ensure the backend is running.".format(app.BACKEND_URL))
    patch_streamlit.button.assert_called_with("Retry Connection")

def test_backend_down_retry_reruns(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: False)
    patch_streamlit.button.return_value = True
    patch_streamlit.stop.side_effect = Exception("stop called")
    patch_streamlit.rerun.side_effect = Exception("rerun called")
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    try:
        importlib.reload(app)
    except Exception as e:
        assert "rerun called" in str(e)
    patch_streamlit.rerun.assert_called_once()

def test_upload_and_index_documents_success(monkeypatch, patch_streamlit, patch_requests):
    # Simulate backend up
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = [
        MagicMock(name="file1", spec=["name", "getvalue", "type"], name="doc1.pdf", getvalue=lambda: b"abc", type="pdf"),
        MagicMock(name="file2", spec=["name", "getvalue", "type"], name="doc2.txt", getvalue=lambda: b"def", type="txt"),
    ]
    patch_streamlit.button.side_effect = [True, False, False]
    patch_requests.post.return_value.status_code = 200
    patch_requests.post.return_value.json.return_value = {
        "message": "Indexed!",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 3},
            {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    }
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.expander.return_value.__enter__.return_value = None
    patch_streamlit.expander.return_value.__exit__.return_value = None
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_requests.post.assert_any_call(
        f"{app.BACKEND_URL}/upload",
        files=[
            ("files", ("doc1.pdf", b"abc", "pdf")),
            ("files", ("doc2.txt", b"def", "txt"))
        ]
    )
    patch_streamlit.success.assert_called_with("Indexed!")
    patch_streamlit.expander.assert_called_with("Extraction Summary")

def test_upload_and_index_documents_no_files(monkeypatch, patch_streamlit):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.side_effect = [True, False, False]
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.warning.assert_any_call("Please upload at least one document.")

def test_upload_and_index_documents_backend_error(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = [
        MagicMock(name="file1", spec=["name", "getvalue", "type"], name="doc1.pdf", getvalue=lambda: b"abc", type="pdf")
    ]
    patch_streamlit.button.side_effect = [True, False, False]
    patch_requests.post.return_value.status_code = 400
    patch_requests.post.return_value.text = "Bad request"
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.error.assert_any_call("Error: Bad request")

def test_upload_and_index_documents_request_exception(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = [
        MagicMock(name="file1", spec=["name", "getvalue", "type"], name="doc1.pdf", getvalue=lambda: b"abc", type="pdf")
    ]
    patch_streamlit.button.side_effect = [True, False, False]
    patch_requests.post.side_effect = Exception("network fail")
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.error.assert_any_call("Request failed: network fail")

def test_chat_qa_happy_path(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.side_effect = [False, False, False]
    # Simulate chat input
    patch_streamlit.chat_input.return_value = "What is the rate?"
    patch_streamlit.session_state = {"messages": []}
    patch_requests.post.return_value.status_code = 200
    patch_requests.post.return_value.json.return_value = {
        "answer": "The rate is $1000.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Relevant text"}
        ]
    }
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.chat_message.return_value.__enter__.return_value = None
    patch_streamlit.chat_message.return_value.__exit__.return_value = None
    patch_streamlit.expander.return_value.__enter__.return_value = None
    patch_streamlit.expander.return_value.__exit__.return_value = None
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_requests.post.assert_any_call(
        f"{app.BACKEND_URL}/ask",
        json={"question": "What is the rate?", "chat_history": []}
    )
    patch_streamlit.markdown.assert_any_call("The rate is $1000.")
    patch_streamlit.caption.assert_any_call("Confidence Score: 0.98")
    patch_streamlit.expander.assert_any_call("View Sources")

def test_chat_qa_backend_error(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.side_effect = [False, False, False]
    patch_streamlit.chat_input.return_value = "What is the rate?"
    patch_streamlit.session_state = {"messages": []}
    patch_requests.post.return_value.status_code = 400
    patch_requests.post.return_value.text = "Bad request"
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.chat_message.return_value.__enter__.return_value = None
    patch_streamlit.chat_message.return_value.__exit__.return_value = None
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.error.assert_any_call("Error: Bad request")

def test_chat_qa_request_exception(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.side_effect = [False, False, False]
    patch_streamlit.chat_input.return_value = "What is the rate?"
    patch_streamlit.session_state = {"messages": []}
    patch_requests.post.side_effect = Exception("network fail")
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.chat_message.return_value.__enter__.return_value = None
    patch_streamlit.chat_message.return_value.__exit__.return_value = None
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.error.assert_any_call("Request failed: network fail")

def test_data_extraction_success(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.side_effect = [[], [], MagicMock(name="extract_file", spec=["name", "getvalue", "type"], name="doc1.pdf", getvalue=lambda: b"abc", type="pdf")]
    patch_streamlit.button.side_effect = [False, False, True]
    patch_requests.post.return_value.status_code = 200
    patch_requests.post.return_value.json.return_value = {
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
            "equipment_type": "Flatbed"
        }
    }
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.columns.return_value = (MagicMock(), MagicMock())
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    patch_streamlit.expander.return_value.__enter__.return_value = None
    patch_streamlit.expander.return_value.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.success.assert_any_call("Extraction complete!")
    patch_streamlit.json.assert_any_call(patch_requests.post.return_value.json.return_value["data"])

def test_data_extraction_backend_error(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.side_effect = [[], [], MagicMock(name="extract_file", spec=["name", "getvalue", "type"], name="doc1.pdf", getvalue=lambda: b"abc", type="pdf")]
    patch_streamlit.button.side_effect = [False, False, True]
    patch_requests.post.return_value.status_code = 400
    patch_requests.post.return_value.text = "Bad request"
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.columns.return_value = (MagicMock(), MagicMock())
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.error.assert_any_call("Error: Bad request")

def test_data_extraction_request_exception(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.side_effect = [[], [], MagicMock(name="extract_file", spec=["name", "getvalue", "type"], name="doc1.pdf", getvalue=lambda: b"abc", type="pdf")]
    patch_streamlit.button.side_effect = [False, False, True]
    patch_requests.post.side_effect = Exception("network fail")
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.columns.return_value = (MagicMock(), MagicMock())
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.error.assert_any_call("Request failed: network fail")

def test_data_extraction_no_file(monkeypatch, patch_streamlit):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.side_effect = [[], [], None]
    patch_streamlit.button.side_effect = [False, False, True]
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.warning.assert_any_call("Please upload a document.")

def test_data_extraction_missing_fields(monkeypatch, patch_streamlit, patch_requests):
    monkeypatch.setattr(app, "check_backend", lambda: True)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.side_effect = [[], [], MagicMock(name="extract_file", spec=["name", "getvalue", "type"], name="doc1.pdf", getvalue=lambda: b"abc", type="pdf")]
    patch_streamlit.button.side_effect = [False, False, True]
    patch_requests.post.return_value.status_code = 200
    patch_requests.post.return_value.json.return_value = {
        "data": {
            # All fields missing
        }
    }
    patch_streamlit.spinner.return_value.__enter__.return_value = None
    patch_streamlit.spinner.return_value.__exit__.return_value = None
    patch_streamlit.columns.return_value = (MagicMock(), MagicMock())
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    patch_streamlit.expander.return_value.__enter__.return_value = None
    patch_streamlit.expander.return_value.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.success.assert_any_call("Extraction complete!")
    patch_streamlit.json.assert_any_call({})

def test_sidebar_about_section(patch_streamlit):
    patch_streamlit.sidebar.__enter__.return_value = None
    patch_streamlit.sidebar.__exit__.return_value = None
    import importlib
    importlib.reload(app)
    patch_streamlit.markdown.assert_any_call("""
    ### About
    This assistant helps you interact with logistics documents using RAG (Retrieval-Augmented Generation).
    
    **Features:**
    - Document Q&A
    - Structured Data Extraction
    - Confidence Scoring
    - Grounded Answers
    """)
