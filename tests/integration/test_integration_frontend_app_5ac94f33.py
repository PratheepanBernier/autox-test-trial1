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
import io

import frontend.app

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit UI functions to no-op or record calls
    st_mock = MagicMock()
    st_mock.sidebar = MagicMock()
    st_mock.sidebar.header = MagicMock()
    st_mock.sidebar.divider = MagicMock()
    st_mock.sidebar.markdown = MagicMock()
    st_mock.set_page_config = MagicMock()
    st_mock.title = MagicMock()
    st_mock.header = MagicMock()
    st_mock.write = MagicMock()
    st_mock.file_uploader = MagicMock()
    st_mock.button = MagicMock()
    st_mock.spinner = MagicMock()
    st_mock.success = MagicMock()
    st_mock.warning = MagicMock()
    st_mock.error = MagicMock()
    st_mock.tabs = MagicMock()
    st_mock.session_state = {}
    st_mock.chat_message = MagicMock()
    st_mock.markdown = MagicMock()
    st_mock.caption = MagicMock()
    st_mock.expander = MagicMock()
    st_mock.chat_input = MagicMock()
    st_mock.columns = MagicMock()
    st_mock.json = MagicMock()
    st_mock.stop = MagicMock()
    st_mock.rerun = MagicMock()
    st_mock.divider = MagicMock()
    st_mock.subheader = MagicMock()
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    st_mock.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.chat_message.return_value.__enter__.return_value = None

    monkeypatch.setattr("frontend.app.st", st_mock)
    return st_mock

@pytest.fixture
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    get_mock = MagicMock()
    post_mock = MagicMock()
    monkeypatch.setattr("frontend.app.requests.get", get_mock)
    monkeypatch.setattr("frontend.app.requests.post", post_mock)
    return get_mock, post_mock

@pytest.fixture
def patch_env(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://mocked-backend:9999")

def make_file_mock(name, content, type_):
    file_mock = MagicMock()
    file_mock.name = name
    file_mock.getvalue.return_value = content
    file_mock.type = type_
    return file_mock

def test_check_backend_happy_path(patch_streamlit, patch_requests, patch_env):
    get_mock, _ = patch_requests
    get_mock.return_value.status_code = 200

    # Should return True when backend responds 200
    assert frontend.app.check_backend() is True
    get_mock.assert_called_once_with("http://mocked-backend:9999/ping")

def test_check_backend_unreachable(patch_streamlit, patch_requests, patch_env):
    get_mock, _ = patch_requests
    get_mock.side_effect = Exception("Connection error")

    # Should return False on exception
    assert frontend.app.check_backend() is False

def test_check_backend_non_200(patch_streamlit, patch_requests, patch_env):
    get_mock, _ = patch_requests
    get_mock.return_value.status_code = 404

    # Should return False if status is not 200
    assert frontend.app.check_backend() is False

def test_upload_and_index_documents_success(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    # Simulate uploaded files
    file1 = make_file_mock("doc1.pdf", b"pdfcontent", "application/pdf")
    file2 = make_file_mock("doc2.txt", b"txtcontent", "text/plain")
    patch_streamlit.file_uploader.return_value = [file1, file2]
    patch_streamlit.button.side_effect = lambda label=None: label == "Process & Index Documents"

    # Simulate backend response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "message": "Indexed 2 documents.",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
            {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    }
    post_mock.return_value = response_mock

    # Simulate tab selection
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    # Run the module (simulate Streamlit run)
    import importlib
    importlib.reload(frontend.app)

    post_mock.assert_any_call(
        "http://mocked-backend:9999/upload",
        files=[
            ("files", ("doc1.pdf", b"pdfcontent", "application/pdf")),
            ("files", ("doc2.txt", b"txtcontent", "text/plain")),
        ]
    )
    patch_streamlit.success.assert_any_call("Indexed 2 documents.")

def test_upload_and_index_documents_no_files(patch_streamlit, patch_requests, patch_env):
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.side_effect = lambda label=None: label == "Process & Index Documents"
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.warning.assert_any_call("Please upload at least one document.")

def test_upload_and_index_documents_backend_error(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    file1 = make_file_mock("doc1.pdf", b"pdfcontent", "application/pdf")
    patch_streamlit.file_uploader.return_value = [file1]
    patch_streamlit.button.side_effect = lambda label=None: label == "Process & Index Documents"
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]

    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Server Error"
    post_mock.return_value = response_mock

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.error.assert_any_call("Error: Internal Server Error")

def test_upload_and_index_documents_request_exception(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    file1 = make_file_mock("doc1.pdf", b"pdfcontent", "application/pdf")
    patch_streamlit.file_uploader.return_value = [file1]
    patch_streamlit.button.side_effect = lambda label=None: label == "Process & Index Documents"
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]

    post_mock.side_effect = Exception("Timeout")

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.error.assert_any_call("Request failed: Timeout")

def test_upload_and_index_documents_with_errors_in_response(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    file1 = make_file_mock("doc1.pdf", b"pdfcontent", "application/pdf")
    patch_streamlit.file_uploader.return_value = [file1]
    patch_streamlit.button.side_effect = lambda label=None: label == "Process & Index Documents"
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "message": "Indexed 1 document.",
        "extractions": [],
        "errors": ["doc1.pdf: Failed to extract text."]
    }
    post_mock.return_value = response_mock

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.warning.assert_any_call("doc1.pdf: Failed to extract text.")

def test_backend_unreachable_error_message_and_retry(patch_streamlit, patch_requests, patch_env):
    # Simulate backend down
    get_mock, _ = patch_requests
    get_mock.side_effect = Exception("Connection error")
    patch_streamlit.button.side_effect = lambda label=None: label == "Retry Connection"
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.error.assert_any_call("⚠️ Cannot connect to backend at http://mocked-backend:9999. Please ensure the backend is running.")
    patch_streamlit.button.assert_any_call("Retry Connection")
    patch_streamlit.stop.assert_called()

def test_question_answering_happy_path(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.chat_input.return_value = "What is the agreed rate for this shipment?"
    patch_streamlit.session_state.clear()
    patch_streamlit.session_state["messages"] = []

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
        ]
    }
    post_mock.return_value = response_mock

    import importlib
    importlib.reload(frontend.app)

    post_mock.assert_any_call(
        "http://mocked-backend:9999/ask",
        json={"question": "What is the agreed rate for this shipment?", "chat_history": []}
    )
    patch_streamlit.markdown.assert_any_call("The agreed rate is $1200.")
    patch_streamlit.caption.assert_any_call("Confidence Score: 0.98")

def test_question_answering_backend_error(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.chat_input.return_value = "What is the agreed rate for this shipment?"
    patch_streamlit.session_state.clear()
    patch_streamlit.session_state["messages"] = []

    response_mock = MagicMock()
    response_mock.status_code = 400
    response_mock.text = "Bad Request"
    post_mock.return_value = response_mock

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.error.assert_any_call("Error: Bad Request")

def test_question_answering_request_exception(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.chat_input.return_value = "What is the agreed rate for this shipment?"
    patch_streamlit.session_state.clear()
    patch_streamlit.session_state["messages"] = []

    post_mock.side_effect = Exception("Timeout")

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.error.assert_any_call("Request failed: Timeout")

def test_data_extraction_success(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    file1 = make_file_mock("doc1.pdf", b"pdfcontent", "application/pdf")
    patch_streamlit.file_uploader.side_effect = lambda *a, **kw: file1 if kw.get("key") == "extraction_uploader" else []
    patch_streamlit.button.side_effect = lambda label=None: label == "Run Extraction"

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Acme Inc.",
            "consignee": "Beta LLC",
            "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
            "driver": {"driver_name": "John Doe", "truck_number": "TRK001"},
            "pickup": {"city": "Dallas", "state": "TX"},
            "shipping_date": "2023-01-01",
            "drop": {"city": "Houston", "state": "TX"},
            "delivery_date": "2023-01-02",
            "rate_info": {"total_rate": "1200", "currency": "USD"},
            "equipment_type": "Van"
        }
    }
    post_mock.return_value = response_mock

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.success.assert_any_call("Extraction complete!")
    patch_streamlit.json.assert_any_call(response_mock.json.return_value["data"])

def test_data_extraction_backend_error(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    file1 = make_file_mock("doc1.pdf", b"pdfcontent", "application/pdf")
    patch_streamlit.file_uploader.side_effect = lambda *a, **kw: file1 if kw.get("key") == "extraction_uploader" else []
    patch_streamlit.button.side_effect = lambda label=None: label == "Run Extraction"

    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Server Error"
    post_mock.return_value = response_mock

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.error.assert_any_call("Error: Internal Server Error")

def test_data_extraction_request_exception(patch_streamlit, patch_requests, patch_env):
    _, post_mock = patch_requests
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    file1 = make_file_mock("doc1.pdf", b"pdfcontent", "application/pdf")
    patch_streamlit.file_uploader.side_effect = lambda *a, **kw: file1 if kw.get("key") == "extraction_uploader" else []
    patch_streamlit.button.side_effect = lambda label=None: label == "Run Extraction"

    post_mock.side_effect = Exception("Timeout")

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.error.assert_any_call("Request failed: Timeout")

def test_data_extraction_no_file_uploaded(patch_streamlit, patch_requests, patch_env):
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.side_effect = lambda *a, **kw: None
    patch_streamlit.button.side_effect = lambda label=None: label == "Run Extraction"

    import importlib
    importlib.reload(frontend.app)

    patch_streamlit.warning.assert_any_call("Please upload a document.")

def test_sidebar_backend_url_env_default(monkeypatch, patch_streamlit):
    # Remove BACKEND_URL env to test default
    monkeypatch.delenv("BACKEND_URL", raising=False)
    patch_streamlit.text_input.return_value = "http://localhost:8000"
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    import importlib
    importlib.reload(frontend.app)
    patch_streamlit.text_input.assert_any_call(
        "Backend API URL",
        value="http://localhost:8000",
        help="The URL of the FastAPI backend."
    )

def test_sidebar_backend_url_env_override(monkeypatch, patch_streamlit):
    monkeypatch.setenv("BACKEND_URL", "http://custom-backend:1234")
    patch_streamlit.text_input.return_value = "http://custom-backend:1234"
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    import importlib
    importlib.reload(frontend.app)
    patch_streamlit.text_input.assert_any_call(
        "Backend API URL",
        value="http://custom-backend:1234",
        help="The URL of the FastAPI backend."
    )
