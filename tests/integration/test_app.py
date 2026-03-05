# source_hash: 9ee607a3d8da4254
import pytest
import builtins
import types
import io

from unittest.mock import patch, MagicMock

# Patch streamlit and requests globally for all tests
@pytest.fixture(autouse=True)
def patch_streamlit_and_requests(monkeypatch):
    # Mock streamlit API
    st_mock = MagicMock()
    st_mock.sidebar = MagicMock()
    st_mock.sidebar.header = MagicMock()
    st_mock.sidebar.text_input = MagicMock(return_value="http://mocked-backend:8000")
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
    st_mock.chat_input = MagicMock()
    st_mock.markdown = MagicMock()
    st_mock.caption = MagicMock()
    st_mock.expander = MagicMock()
    st_mock.columns = MagicMock()
    st_mock.json = MagicMock()
    st_mock.stop = MagicMock()
    st_mock.rerun = MagicMock()
    st_mock.subheader = MagicMock()
    st_mock.columns = MagicMock(return_value=(MagicMock(), MagicMock()))
    st_mock.session_state = {}
    st_mock.session_state.messages = []
    st_mock.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    st_mock.expander = MagicMock()
    st_mock.expander.__enter__ = lambda s: s
    st_mock.expander.__exit__ = lambda s, exc_type, exc_val, exc_tb: None
    st_mock.spinner = lambda msg: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, e, v, t: None)
    st_mock.chat_message = lambda role: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, e, v, t: None)
    st_mock.columns = lambda n: tuple(MagicMock() for _ in range(n))
    st_mock.session_state = {}

    monkeypatch.setitem(__import__('sys').modules, "streamlit", st_mock)

    # Patch requests
    requests_mock = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, "requests", requests_mock)

    yield st_mock, requests_mock

@pytest.fixture
def reload_app(monkeypatch):
    import importlib
    def _reload():
        if "frontend.app" in __import__('sys').modules:
            del __import__('sys').modules["frontend.app"]
        import frontend.app
        importlib.reload(frontend.app)
        return frontend.app
    return _reload

def make_file_mock(name, content, filetype):
    file_mock = MagicMock()
    file_mock.name = name
    file_mock.getvalue = MagicMock(return_value=content)
    file_mock.type = filetype
    return file_mock

def test_backend_health_check_success(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    # Simulate backend /ping returns 200
    requests_mock.get.return_value.status_code = 200
    reload_app()
    requests_mock.get.assert_called_with("http://mocked-backend:8000/ping")
    # Should not call st_mock.error or st_mock.stop
    assert not st_mock.error.called
    assert not st_mock.stop.called

def test_backend_health_check_failure_and_retry(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    # Simulate backend /ping fails
    requests_mock.get.side_effect = Exception("Connection error")
    st_mock.button.return_value = False
    reload_app()
    st_mock.error.assert_called()
    st_mock.stop.assert_called()

def test_backend_health_check_failure_and_retry_button(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    # Simulate backend /ping fails
    requests_mock.get.side_effect = Exception("Connection error")
    st_mock.button.return_value = True
    reload_app()
    st_mock.rerun.assert_called()

def test_upload_and_index_documents_happy_path(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    # Backend health OK
    requests_mock.get.return_value.status_code = 200
    # Simulate file upload
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    file2 = make_file_mock("doc2.txt", b"TXTDATA", "text/plain")
    st_mock.file_uploader.side_effect = [[file1, file2], None, None]
    st_mock.button.side_effect = [True, False, False]
    # Simulate backend /upload
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
    requests_mock.post.return_value = response_mock
    reload_app()
    requests_mock.post.assert_any_call(
        "http://mocked-backend:8000/upload",
        files=[
            ("files", ("doc1.pdf", b"PDFDATA", "application/pdf")),
            ("files", ("doc2.txt", b"TXTDATA", "text/plain"))
        ]
    )
    st_mock.success.assert_any_call("Indexed 2 documents.")

def test_upload_and_index_documents_no_files(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, None]
    st_mock.button.side_effect = [True, False, False]
    reload_app()
    st_mock.warning.assert_any_call("Please upload at least one document.")

def test_upload_and_index_documents_backend_error(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    st_mock.file_uploader.side_effect = [[file1], None, None]
    st_mock.button.side_effect = [True, False, False]
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Server Error"
    requests_mock.post.return_value = response_mock
    reload_app()
    st_mock.error.assert_any_call("Error: Internal Server Error")

def test_upload_and_index_documents_request_exception(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    st_mock.file_uploader.side_effect = [[file1], None, None]
    st_mock.button.side_effect = [True, False, False]
    requests_mock.post.side_effect = Exception("Timeout")
    reload_app()
    st_mock.error.assert_any_call("Request failed: Timeout")

def test_upload_and_index_documents_with_errors_in_response(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    st_mock.file_uploader.side_effect = [[file1], None, None]
    st_mock.button.side_effect = [True, False, False]
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "message": "Indexed 1 document.",
        "extractions": [],
        "errors": ["doc1.pdf: Failed to extract text."]
    }
    requests_mock.post.return_value = response_mock
    reload_app()
    st_mock.warning.assert_any_call("doc1.pdf: Failed to extract text.")

def test_chat_qa_happy_path(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, None]
    st_mock.button.side_effect = [False, False, False]
    st_mock.chat_input.return_value = "What is the agreed rate?"
    # Simulate backend /ask
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
        ]
    }
    requests_mock.post.return_value = response_mock
    reload_app()
    requests_mock.post.assert_any_call(
        "http://mocked-backend:8000/ask",
        json={"question": "What is the agreed rate?", "chat_history": []}
    )
    st_mock.markdown.assert_any_call("The agreed rate is $1200.")
    st_mock.caption.assert_any_call("Confidence Score: 0.98")

def test_chat_qa_backend_error(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, None]
    st_mock.button.side_effect = [False, False, False]
    st_mock.chat_input.return_value = "What is the agreed rate?"
    response_mock = MagicMock()
    response_mock.status_code = 400
    response_mock.text = "Bad Request"
    requests_mock.post.return_value = response_mock
    reload_app()
    st_mock.error.assert_any_call("Error: Bad Request")

def test_chat_qa_request_exception(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, None]
    st_mock.button.side_effect = [False, False, False]
    st_mock.chat_input.return_value = "What is the agreed rate?"
    requests_mock.post.side_effect = Exception("Timeout")
    reload_app()
    st_mock.error.assert_any_call("Request failed: Timeout")

def test_data_extraction_happy_path(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, make_file_mock("doc3.pdf", b"PDFDATA", "application/pdf")]
    st_mock.button.side_effect = [False, False, True]
    # Simulate backend /extract
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
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
    requests_mock.post.return_value = response_mock
    reload_app()
    st_mock.success.assert_any_call("Extraction complete!")
    st_mock.json.assert_any_call(response_mock.json.return_value["data"])

def test_data_extraction_no_file_uploaded(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, None]
    st_mock.button.side_effect = [False, False, True]
    reload_app()
    st_mock.warning.assert_any_call("Please upload a document.")

def test_data_extraction_backend_error(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, make_file_mock("doc3.pdf", b"PDFDATA", "application/pdf")]
    st_mock.button.side_effect = [False, False, True]
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Server Error"
    requests_mock.post.return_value = response_mock
    reload_app()
    st_mock.error.assert_any_call("Error: Internal Server Error")

def test_data_extraction_request_exception(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, make_file_mock("doc3.pdf", b"PDFDATA", "application/pdf")]
    st_mock.button.side_effect = [False, False, True]
    requests_mock.post.side_effect = Exception("Timeout")
    reload_app()
    st_mock.error.assert_any_call("Request failed: Timeout")

def test_data_extraction_missing_fields(patch_streamlit_and_requests, reload_app):
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    st_mock.file_uploader.side_effect = [[], None, make_file_mock("doc3.pdf", b"PDFDATA", "application/pdf")]
    st_mock.button.side_effect = [False, False, True]
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "data": {
            # Only a few fields present
            "reference_id": "REF123",
            "pickup": {"city": "Dallas", "state": "TX"},
            "drop": {"city": "Houston", "state": "TX"},
        }
    }
    requests_mock.post.return_value = response_mock
    reload_app()
    st_mock.success.assert_any_call("Extraction complete!")
    st_mock.json.assert_any_call(response_mock.json.return_value["data"])

def test_equivalent_paths_upload_and_extract(patch_streamlit_and_requests, reload_app):
    """
    Reconciliation: Uploading a file and extracting it directly should yield compatible outputs.
    """
    st_mock, requests_mock = patch_streamlit_and_requests
    requests_mock.get.return_value.status_code = 200
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    # Upload path
    st_mock.file_uploader.side_effect = [[file1], None, file1]
    st_mock.button.side_effect = [True, False, True]
    upload_response = MagicMock()
    upload_response.status_code = 200
    upload_response.json.return_value = {
        "message": "Indexed 1 document.",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5}
        ],
        "errors": []
    }
    extract_response = MagicMock()
    extract_response.status_code = 200
    extract_response.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "pickup": {"city": "Dallas", "state": "TX"},
            "drop": {"city": "Houston", "state": "TX"},
        }
    }
    requests_mock.post.side_effect = [upload_response, extract_response]
    reload_app()
    # Both upload and extract should be called
    assert requests_mock.post.call_count >= 2
    # Extraction summary and extraction complete should both be called
    st_mock.success.assert_any_call("Indexed 1 document.")
    st_mock.success.assert_any_call("Extraction complete!")
