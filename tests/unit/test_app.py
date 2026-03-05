# source_hash: 9ee607a3d8da4254
import pytest
from unittest.mock import patch, MagicMock, call

# Patch streamlit and requests globally for all tests
@pytest.fixture(autouse=True)
def patch_streamlit_and_requests(monkeypatch):
    # Patch streamlit
    st_mock = MagicMock()
    monkeypatch.setattr("frontend.app.st", st_mock)
    # Patch requests
    requests_mock = MagicMock()
    monkeypatch.setattr("frontend.app.requests", requests_mock)
    # Patch os
    os_mock = MagicMock()
    monkeypatch.setattr("frontend.app.os", os_mock)
    yield st_mock, requests_mock, os_mock

def _setup_sidebar(st_mock, os_mock, backend_url="http://localhost:8000"):
    # Simulate sidebar config
    os_mock.getenv.return_value = backend_url
    st_mock.sidebar.__enter__.return_value = None
    st_mock.text_input.return_value = backend_url
    st_mock.divider.return_value = None
    st_mock.markdown.return_value = None
    st_mock.header.return_value = None

def _setup_tabs(st_mock):
    # Simulate tabs
    tab_mocks = [MagicMock(), MagicMock(), MagicMock()]
    st_mock.tabs.return_value = tab_mocks
    return tab_mocks

def _simulate_file(name="doc.pdf", content=b"abc", filetype="application/pdf"):
    file_mock = MagicMock()
    file_mock.name = name
    file_mock.getvalue.return_value = content
    file_mock.type = filetype
    return file_mock

def test_check_backend_happy_path(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200

    # Import after patching
    import frontend.app as app

    requests_mock.get.assert_called_with("http://localhost:8000/ping")
    assert app.check_backend() is True

def test_check_backend_failure_returns_false(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.side_effect = Exception("network error")

    import frontend.app as app

    assert app.check_backend() is False

def test_backend_down_shows_error_and_stop(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 500
    st_mock.button.return_value = False

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.error.assert_called()
    st_mock.stop.assert_called()

def test_backend_down_retry_reruns(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 500
    st_mock.button.return_value = True

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.rerun.assert_called()

def test_upload_and_index_documents_success(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    tab_mocks = _setup_tabs(st_mock)
    # Simulate file upload
    file1 = _simulate_file("a.pdf", b"abc", "application/pdf")
    file2 = _simulate_file("b.txt", b"def", "text/plain")
    st_mock.file_uploader.return_value = [file1, file2]
    st_mock.button.side_effect = [True, False, False]  # Only first tab's button pressed
    # Simulate backend response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "message": "Indexed 2 docs",
        "extractions": [
            {"filename": "a.pdf", "structured_data_extracted": True, "text_chunks": 5},
            {"filename": "b.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    }
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.success.assert_any_call("Indexed 2 docs")
    st_mock.expander.assert_called_with("Extraction Summary")
    st_mock.write.assert_any_call("✅ **a.pdf**: 5 chunks")
    st_mock.write.assert_any_call("❌ **b.txt**: 2 chunks")

def test_upload_and_index_documents_with_errors(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    file1 = _simulate_file("a.pdf", b"abc", "application/pdf")
    st_mock.file_uploader.return_value = [file1]
    st_mock.button.side_effect = [True, False, False]
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "message": "Indexed 1 doc",
        "extractions": [],
        "errors": ["File corrupted"]
    }
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.success.assert_any_call("Indexed 1 doc")
    st_mock.warning.assert_any_call("File corrupted")

def test_upload_and_index_documents_backend_error(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    file1 = _simulate_file("a.pdf", b"abc", "application/pdf")
    st_mock.file_uploader.return_value = [file1]
    st_mock.button.side_effect = [True, False, False]
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Error"
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.error.assert_any_call("Error: Internal Error")

def test_upload_and_index_documents_request_exception(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    file1 = _simulate_file("a.pdf", b"abc", "application/pdf")
    st_mock.file_uploader.return_value = [file1]
    st_mock.button.side_effect = [True, False, False]
    requests_mock.post.side_effect = Exception("Timeout")

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.error.assert_any_call("Request failed: Timeout")

def test_upload_and_index_documents_no_files_warning(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    st_mock.file_uploader.return_value = []
    st_mock.button.side_effect = [True, False, False]

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.warning.assert_any_call("Please upload at least one document.")

def test_chat_qa_happy_path(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    # Simulate chat input
    st_mock.session_state = {}
    st_mock.chat_input.return_value = "What is the rate?"
    st_mock.button.side_effect = [False, True, False]
    # Simulate backend response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "answer": "The rate is $1000.",
        "confidence_score": 0.95,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Relevant text"}
        ]
    }
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.markdown.assert_any_call("The rate is $1000.")
    st_mock.caption.assert_any_call("Confidence Score: 0.95")
    st_mock.expander.assert_any_call("View Sources")
    st_mock.caption.assert_any_call("Source: doc1.pdf")
    st_mock.write.assert_any_call("Relevant text")

def test_chat_qa_backend_error(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    st_mock.session_state = {}
    st_mock.chat_input.return_value = "What is the rate?"
    st_mock.button.side_effect = [False, True, False]
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Backend error"
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.error.assert_any_call("Error: Backend error")

def test_chat_qa_request_exception(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    st_mock.session_state = {}
    st_mock.chat_input.return_value = "What is the rate?"
    st_mock.button.side_effect = [False, True, False]
    requests_mock.post.side_effect = Exception("Timeout")

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.error.assert_any_call("Request failed: Timeout")

def test_data_extraction_success(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    extract_file = _simulate_file("extract.pdf", b"xyz", "application/pdf")
    st_mock.file_uploader.side_effect = [[], [], extract_file]
    st_mock.button.side_effect = [False, False, True]
    # Simulate backend response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Shipper Inc",
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
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.success.assert_any_call("Extraction complete!")
    st_mock.subheader.assert_any_call("IDs & Parties")
    st_mock.subheader.assert_any_call("Carrier & Driver")
    st_mock.subheader.assert_any_call("Stops & Dates")
    st_mock.subheader.assert_any_call("Rates & Equipment")
    st_mock.json.assert_any_call(response_mock.json.return_value["data"])

def test_data_extraction_backend_error(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    extract_file = _simulate_file("extract.pdf", b"xyz", "application/pdf")
    st_mock.file_uploader.side_effect = [[], [], extract_file]
    st_mock.button.side_effect = [False, False, True]
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Extraction failed"
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.error.assert_any_call("Error: Extraction failed")

def test_data_extraction_request_exception(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    extract_file = _simulate_file("extract.pdf", b"xyz", "application/pdf")
    st_mock.file_uploader.side_effect = [[], [], extract_file]
    st_mock.button.side_effect = [False, False, True]
    requests_mock.post.side_effect = Exception("Timeout")

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.error.assert_any_call("Request failed: Timeout")

def test_data_extraction_no_file_warning(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    st_mock.file_uploader.side_effect = [[], [], None]
    st_mock.button.side_effect = [False, False, True]

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.warning.assert_any_call("Please upload a document.")

def test_data_extraction_handles_missing_fields(patch_streamlit_and_requests):
    st_mock, requests_mock, os_mock = patch_streamlit_and_requests
    _setup_sidebar(st_mock, os_mock)
    requests_mock.get.return_value.status_code = 200
    _setup_tabs(st_mock)
    extract_file = _simulate_file("extract.pdf", b"xyz", "application/pdf")
    st_mock.file_uploader.side_effect = [[], [], extract_file]
    st_mock.button.side_effect = [False, False, True]
    # Simulate backend response with missing fields
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "data": {
            "reference_id": None,
            "carrier": None,
            "driver": None,
            "pickup": None,
            "drop": None,
            "rate_info": None
        }
    }
    requests_mock.post.return_value = response_mock

    import importlib
    import frontend.app as app
    importlib.reload(app)

    st_mock.write.assert_any_call("**Reference ID:** N/A")
    st_mock.write.assert_any_call("**Carrier:** N/A")
    st_mock.write.assert_any_call("**Driver:** N/A")
    st_mock.write.assert_any_call("**Pickup:** N/A, N/A")
    st_mock.write.assert_any_call("**Drop:** N/A, N/A")
    st_mock.write.assert_any_call("**Total Rate:** N/A ")
    st_mock.write.assert_any_call("**Equipment:** N/A")
