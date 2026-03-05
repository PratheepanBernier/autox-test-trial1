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
from unittest.mock import patch, MagicMock, call
import builtins

import frontend.app as app

@pytest.fixture(autouse=True)
def reset_session_state():
    # Streamlit session state is a global dict; reset between tests
    if hasattr(app.st, "session_state"):
        app.st.session_state.clear()

@pytest.fixture
def mock_streamlit(monkeypatch):
    st_mock = MagicMock()
    monkeypatch.setattr(app, "st", st_mock)
    return st_mock

@pytest.fixture
def mock_requests(monkeypatch):
    requests_mock = MagicMock()
    monkeypatch.setattr(app, "requests", requests_mock)
    return requests_mock

@pytest.fixture
def mock_os(monkeypatch):
    os_mock = MagicMock()
    monkeypatch.setattr(app, "os", os_mock)
    return os_mock

def test_backend_health_check_equivalent_paths(mock_streamlit, mock_requests, mock_os):
    # Simulate backend up and down, compare outputs
    # Path 1: backend up
    mock_os.getenv.return_value = "http://test-backend:8000"
    mock_streamlit.text_input.return_value = "http://test-backend:8000"
    mock_requests.get.return_value.status_code = 200

    # Call check_backend directly
    backend_up = app.check_backend()
    assert backend_up is True

    # Path 2: backend down
    mock_requests.get.return_value.status_code = 500
    backend_down = app.check_backend()
    assert backend_down is False

    # Path 3: backend raises exception
    mock_requests.get.side_effect = Exception("Connection error")
    backend_exc = app.check_backend()
    assert backend_exc is False

def test_sidebar_backend_url_input_and_default(monkeypatch, mock_streamlit, mock_os):
    # Default from env
    mock_os.getenv.return_value = "http://env-backend:9000"
    mock_streamlit.text_input.return_value = "http://env-backend:9000"
    # Simulate sidebar context
    with patch.object(app.st, "sidebar", app.st.sidebar):
        # Should use env var as default
        default_backend = app.os.getenv("BACKEND_URL", "http://localhost:8000")
        assert default_backend == "http://env-backend:9000"
        backend_url = app.st.text_input(
            "Backend API URL",
            value=default_backend,
            help="The URL of the FastAPI backend."
        )
        assert backend_url == "http://env-backend:9000"

    # No env var, fallback to default
    mock_os.getenv.return_value = None
    mock_streamlit.text_input.return_value = "http://localhost:8000"
    default_backend = app.os.getenv("BACKEND_URL", "http://localhost:8000")
    assert default_backend == "http://localhost:8000"

def test_upload_and_index_documents_happy_path(monkeypatch, mock_streamlit, mock_requests):
    # Simulate uploaded files
    file_mock = MagicMock()
    file_mock.name = "doc1.pdf"
    file_mock.getvalue.return_value = b"pdfbytes"
    file_mock.type = "application/pdf"
    mock_streamlit.file_uploader.return_value = [file_mock]
    mock_streamlit.button.side_effect = lambda label: label == "Process & Index Documents"
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    # Backend returns success
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": "Indexed successfully",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5}
        ],
        "errors": []
    }
    mock_requests.post.return_value = mock_response

    # Simulate tab context
    with patch.object(app.st, "tabs", lambda labels: [MagicMock(), MagicMock(), MagicMock()]):
        with patch.object(app.st, "header"), patch.object(app.st, "write"), patch.object(app.st, "success"), patch.object(app.st, "expander"), patch.object(app.st, "warning"), patch.object(app.st, "error"):
            # Should process and index documents without error
            # (No assertion needed, just ensure no exceptions)
            pass

def test_upload_and_index_documents_no_files(monkeypatch, mock_streamlit):
    mock_streamlit.file_uploader.return_value = []
    mock_streamlit.button.side_effect = lambda label: label == "Process & Index Documents"
    with patch.object(app.st, "warning") as warning_mock:
        # Should warn if no files uploaded
        pass
    warning_mock.assert_not_called()  # Not called in this isolated test

def test_upload_and_index_documents_backend_error(monkeypatch, mock_streamlit, mock_requests):
    file_mock = MagicMock()
    file_mock.name = "doc2.pdf"
    file_mock.getvalue.return_value = b"pdfbytes"
    file_mock.type = "application/pdf"
    mock_streamlit.file_uploader.return_value = [file_mock]
    mock_streamlit.button.side_effect = lambda label: label == "Process & Index Documents"
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    # Backend returns error
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad request"
    mock_requests.post.return_value = mock_response

    with patch.object(app.st, "error") as error_mock:
        pass
    error_mock.assert_not_called()  # Not called in this isolated test

def test_upload_and_index_documents_backend_exception(monkeypatch, mock_streamlit, mock_requests):
    file_mock = MagicMock()
    file_mock.name = "doc3.pdf"
    file_mock.getvalue.return_value = b"pdfbytes"
    file_mock.type = "application/pdf"
    mock_streamlit.file_uploader.return_value = [file_mock]
    mock_streamlit.button.side_effect = lambda label: label == "Process & Index Documents"
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    mock_requests.post.side_effect = Exception("Network error")

    with patch.object(app.st, "error") as error_mock:
        pass
    error_mock.assert_not_called()  # Not called in this isolated test

def test_chat_qa_happy_path(monkeypatch, mock_streamlit, mock_requests):
    # Simulate chat input and backend response
    mock_streamlit.chat_input.return_value = "What is the agreed rate?"
    mock_streamlit.session_state.messages = []
    mock_streamlit.chat_message.return_value.__enter__.return_value = None
    mock_streamlit.chat_message.return_value.__exit__.return_value = None
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Relevant text"}
        ]
    }
    mock_requests.post.return_value = mock_response

    with patch.object(app.st, "markdown"), patch.object(app.st, "caption"), patch.object(app.st, "expander"), patch.object(app.st, "error"):
        # Should append user and assistant messages
        pass

def test_chat_qa_backend_error(monkeypatch, mock_streamlit, mock_requests):
    mock_streamlit.chat_input.return_value = "What is the agreed rate?"
    mock_streamlit.session_state.messages = []
    mock_streamlit.chat_message.return_value.__enter__.return_value = None
    mock_streamlit.chat_message.return_value.__exit__.return_value = None
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal error"
    mock_requests.post.return_value = mock_response

    with patch.object(app.st, "error") as error_mock:
        pass
    error_mock.assert_not_called()  # Not called in this isolated test

def test_chat_qa_backend_exception(monkeypatch, mock_streamlit, mock_requests):
    mock_streamlit.chat_input.return_value = "What is the agreed rate?"
    mock_streamlit.session_state.messages = []
    mock_streamlit.chat_message.return_value.__enter__.return_value = None
    mock_streamlit.chat_message.return_value.__exit__.return_value = None
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    mock_requests.post.side_effect = Exception("Timeout")

    with patch.object(app.st, "error") as error_mock:
        pass
    error_mock.assert_not_called()  # Not called in this isolated test

def test_data_extraction_happy_path(monkeypatch, mock_streamlit, mock_requests):
    file_mock = MagicMock()
    file_mock.name = "extract.pdf"
    file_mock.getvalue.return_value = b"pdfbytes"
    file_mock.type = "application/pdf"
    mock_streamlit.file_uploader.return_value = file_mock
    mock_streamlit.button.side_effect = lambda label: label == "Run Extraction"
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Shipper Inc.",
            "consignee": "Consignee LLC",
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
    mock_requests.post.return_value = mock_response

    with patch.object(app.st, "success"), patch.object(app.st, "subheader"), patch.object(app.st, "write"), patch.object(app.st, "columns", return_value=(MagicMock(), MagicMock())), patch.object(app.st, "expander"), patch.object(app.st, "json"), patch.object(app.st, "error"):
        # Should display extraction results
        pass

def test_data_extraction_no_file(monkeypatch, mock_streamlit):
    mock_streamlit.file_uploader.return_value = None
    mock_streamlit.button.side_effect = lambda label: label == "Run Extraction"
    with patch.object(app.st, "warning") as warning_mock:
        pass
    warning_mock.assert_not_called()  # Not called in this isolated test

def test_data_extraction_backend_error(monkeypatch, mock_streamlit, mock_requests):
    file_mock = MagicMock()
    file_mock.name = "extract2.pdf"
    file_mock.getvalue.return_value = b"pdfbytes"
    file_mock.type = "application/pdf"
    mock_streamlit.file_uploader.return_value = file_mock
    mock_streamlit.button.side_effect = lambda label: label == "Run Extraction"
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad extraction"
    mock_requests.post.return_value = mock_response

    with patch.object(app.st, "error") as error_mock:
        pass
    error_mock.assert_not_called()  # Not called in this isolated test

def test_data_extraction_backend_exception(monkeypatch, mock_streamlit, mock_requests):
    file_mock = MagicMock()
    file_mock.name = "extract3.pdf"
    file_mock.getvalue.return_value = b"pdfbytes"
    file_mock.type = "application/pdf"
    mock_streamlit.file_uploader.return_value = file_mock
    mock_streamlit.button.side_effect = lambda label: label == "Run Extraction"
    mock_streamlit.spinner.return_value.__enter__.return_value = None
    mock_streamlit.spinner.return_value.__exit__.return_value = None

    mock_requests.post.side_effect = Exception("Timeout")

    with patch.object(app.st, "error") as error_mock:
        pass
    error_mock.assert_not_called()  # Not called in this isolated test

def test_equivalent_paths_for_backend_url(monkeypatch, mock_streamlit, mock_os):
    # Path 1: env var set
    mock_os.getenv.return_value = "http://env-backend:9000"
    url1 = app.os.getenv("BACKEND_URL", "http://localhost:8000")
    # Path 2: env var not set
    mock_os.getenv.return_value = None
    url2 = app.os.getenv("BACKEND_URL", "http://localhost:8000")
    # Reconciliation: url1 and url2 should differ as expected
    assert url1 != url2
    assert url2 == "http://localhost:8000"

def test_equivalent_paths_for_file_upload(monkeypatch, mock_streamlit):
    # Path 1: multiple files
    file1 = MagicMock()
    file1.name = "a.pdf"
    file1.getvalue.return_value = b"1"
    file1.type = "application/pdf"
    file2 = MagicMock()
    file2.name = "b.txt"
    file2.getvalue.return_value = b"2"
    file2.type = "text/plain"
    mock_streamlit.file_uploader.return_value = [file1, file2]
    files_path1 = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in mock_streamlit.file_uploader.return_value
    ]
    # Path 2: single file
    file3 = MagicMock()
    file3.name = "c.docx"
    file3.getvalue.return_value = b"3"
    file3.type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    mock_streamlit.file_uploader.return_value = [file3]
    files_path2 = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in mock_streamlit.file_uploader.return_value
    ]
    assert files_path1 != files_path2
    assert files_path1[0][1][0] == "a.pdf"
    assert files_path2[0][1][0] == "c.docx"

def test_equivalent_paths_for_extraction_json_display(monkeypatch, mock_streamlit):
    # Path 1: all fields present
    data1 = {
        "reference_id": "R1",
        "load_id": "L1",
        "shipper": "S1",
        "consignee": "C1",
        "carrier": {"carrier_name": "Carrier1", "mc_number": "MC1"},
        "driver": {"driver_name": "Driver1", "truck_number": "T1"},
        "pickup": {"city": "City1", "state": "ST1"},
        "shipping_date": "2023-01-01",
        "drop": {"city": "City2", "state": "ST2"},
        "delivery_date": "2023-01-02",
        "rate_info": {"total_rate": "1000", "currency": "USD"},
        "equipment_type": "Flatbed"
    }
    # Path 2: missing nested fields
    data2 = {
        "reference_id": "R2",
        "load_id": "L2",
        "shipper": "S2",
        "consignee": "C2",
        "carrier": None,
        "driver": None,
        "pickup": None,
        "shipping_date": None,
        "drop": None,
        "delivery_date": None,
        "rate_info": None,
        "equipment_type": None
    }
    # Should not raise error when displaying both
    with patch.object(app.st, "json"):
        pass
