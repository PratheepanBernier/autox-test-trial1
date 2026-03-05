import pytest
import builtins
import types
from unittest.mock import patch, MagicMock, call

# Regression tests for frontend/app.py

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    """Patch streamlit methods to no-op or record calls."""
    st_mock = MagicMock()
    # Patch all used streamlit methods
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
    st_mock.divider = MagicMock()
    st_mock.markdown = MagicMock()
    st_mock.tabs = MagicMock()
    st_mock.session_state = {}
    st_mock.chat_message = MagicMock()
    st_mock.chat_input = MagicMock()
    st_mock.caption = MagicMock()
    st_mock.expander = MagicMock()
    st_mock.columns = MagicMock()
    st_mock.stop = MagicMock(side_effect=Exception("st.stop called"))
    st_mock.rerun = MagicMock(side_effect=Exception("st.rerun called"))
    st_mock.json = MagicMock()
    st_mock.sidebar = MagicMock()
    st_mock.subheader = MagicMock()
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.expander.return_value.__exit__.return_value = None
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    st_mock.sidebar.__enter__.return_value = None
    st_mock.sidebar.__exit__.return_value = None
    st_mock.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    monkeypatch.setitem(__import__('sys').modules, "streamlit", st_mock)
    return st_mock

@pytest.fixture
def patch_requests(monkeypatch):
    """Patch requests.get and requests.post."""
    requests_mock = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, "requests", requests_mock)
    return requests_mock

@pytest.fixture
def patch_os(monkeypatch):
    os_mock = MagicMock()
    os_mock.getenv = MagicMock(return_value="http://localhost:8000")
    monkeypatch.setitem(__import__('sys').modules, "os", os_mock)
    return os_mock

def import_app():
    # Import the app.py module, reload to reset state between tests
    import importlib
    import frontend.app as app
    importlib.reload(app)
    return app

def test_backend_health_happy_path(patch_streamlit, patch_requests, patch_os):
    # Simulate backend /ping healthy
    patch_requests.get.return_value.status_code = 200
    # Simulate all tabs, no file uploads, no chat input
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.return_value = False
    patch_streamlit.chat_input.return_value = None
    import_app()
    # Should check backend and not show error
    assert not patch_streamlit.error.called or all(
        "Cannot connect to backend" not in str(call) for call in patch_streamlit.error.call_args_list
    )
    # Should call set_page_config and title
    patch_streamlit.set_page_config.assert_called_once()
    patch_streamlit.title.assert_called_once()

def test_backend_health_down_shows_error_and_stop(patch_streamlit, patch_requests, patch_os):
    # Simulate backend /ping unhealthy
    patch_requests.get.side_effect = Exception("Connection error")
    patch_streamlit.button.return_value = False
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.chat_input.return_value = None
    with pytest.raises(Exception) as excinfo:
        import_app()
    # Should show error and call st.stop
    patch_streamlit.error.assert_any_call("⚠️ Cannot connect to backend at http://localhost:8000. Please ensure the backend is running.")
    patch_streamlit.stop.assert_called_once()
    assert "st.stop called" in str(excinfo.value)

def test_backend_health_down_retry_calls_rerun(patch_streamlit, patch_requests, patch_os):
    # Simulate backend /ping unhealthy
    patch_requests.get.side_effect = Exception("Connection error")
    patch_streamlit.button.side_effect = [True]  # Simulate user clicks "Retry Connection"
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.chat_input.return_value = None
    with pytest.raises(Exception) as excinfo:
        import_app()
    patch_streamlit.rerun.assert_called_once()
    assert "st.rerun called" in str(excinfo.value)

def test_upload_index_no_files_warns(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Process & Index Documents" in a else False
    patch_streamlit.chat_input.return_value = None
    import_app()
    patch_streamlit.warning.assert_any_call("Please upload at least one document.")

def test_upload_index_successful_upload(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    # Simulate file upload
    fake_file = MagicMock()
    fake_file.name = "test.pdf"
    fake_file.getvalue.return_value = b"pdfbytes"
    fake_file.type = "application/pdf"
    patch_streamlit.file_uploader.return_value = [fake_file]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Process & Index Documents" in a else False
    patch_streamlit.chat_input.return_value = None
    # Simulate backend /upload
    upload_response = MagicMock()
    upload_response.status_code = 200
    upload_response.json.return_value = {
        "message": "Indexed successfully",
        "extractions": [
            {"filename": "test.pdf", "structured_data_extracted": True, "text_chunks": 5}
        ],
        "errors": []
    }
    patch_requests.post.return_value = upload_response
    import_app()
    patch_streamlit.success.assert_any_call("Indexed successfully")
    patch_streamlit.expander.assert_any_call("Extraction Summary")

def test_upload_index_backend_error(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    fake_file = MagicMock()
    fake_file.name = "fail.pdf"
    fake_file.getvalue.return_value = b"fail"
    fake_file.type = "application/pdf"
    patch_streamlit.file_uploader.return_value = [fake_file]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Process & Index Documents" in a else False
    patch_streamlit.chat_input.return_value = None
    upload_response = MagicMock()
    upload_response.status_code = 500
    upload_response.text = "Internal Server Error"
    patch_requests.post.return_value = upload_response
    import_app()
    patch_streamlit.error.assert_any_call("Error: Internal Server Error")

def test_upload_index_request_exception(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    fake_file = MagicMock()
    fake_file.name = "fail.pdf"
    fake_file.getvalue.return_value = b"fail"
    fake_file.type = "application/pdf"
    patch_streamlit.file_uploader.return_value = [fake_file]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Process & Index Documents" in a else False
    patch_streamlit.chat_input.return_value = None
    patch_requests.post.side_effect = Exception("Network down")
    import_app()
    patch_streamlit.error.assert_any_call("Request failed: Network down")

def test_chat_qa_happy_path(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.return_value = False
    # Simulate chat input
    patch_streamlit.chat_input.return_value = "What is the agreed rate?"
    # Simulate backend /ask
    ask_response = MagicMock()
    ask_response.status_code = 200
    ask_response.json.return_value = {
        "answer": "The agreed rate is $1000.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "test.pdf"}, "text": "Rate: $1000"}
        ]
    }
    patch_requests.post.return_value = ask_response
    import_app()
    patch_streamlit.markdown.assert_any_call("The agreed rate is $1000.")
    patch_streamlit.caption.assert_any_call("Confidence Score: 0.98")
    patch_streamlit.expander.assert_any_call("View Sources")

def test_chat_qa_backend_error(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.return_value = False
    patch_streamlit.chat_input.return_value = "What is the agreed rate?"
    ask_response = MagicMock()
    ask_response.status_code = 400
    ask_response.text = "Bad Request"
    patch_requests.post.return_value = ask_response
    import_app()
    patch_streamlit.error.assert_any_call("Error: Bad Request")

def test_chat_qa_request_exception(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.return_value = []
    patch_streamlit.button.return_value = False
    patch_streamlit.chat_input.return_value = "What is the agreed rate?"
    patch_requests.post.side_effect = Exception("Timeout")
    import_app()
    patch_streamlit.error.assert_any_call("Request failed: Timeout")

def test_data_extraction_no_file_warns(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    patch_streamlit.file_uploader.side_effect = [[], None]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Run Extraction" in a else False
    patch_streamlit.chat_input.return_value = None
    import_app()
    patch_streamlit.warning.assert_any_call("Please upload a document.")

def test_data_extraction_success(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    # Simulate file upload for extraction
    fake_file = MagicMock()
    fake_file.name = "extract.pdf"
    fake_file.getvalue.return_value = b"extract"
    fake_file.type = "application/pdf"
    patch_streamlit.file_uploader.side_effect = [[], fake_file]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Run Extraction" in a else False
    patch_streamlit.chat_input.return_value = None
    extract_response = MagicMock()
    extract_response.status_code = 200
    extract_response.json.return_value = {
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
            "rate_info": {"total_rate": "1000", "currency": "USD"},
            "equipment_type": "Van"
        }
    }
    patch_requests.post.return_value = extract_response
    import_app()
    patch_streamlit.success.assert_any_call("Extraction complete!")
    patch_streamlit.json.assert_called()

def test_data_extraction_backend_error(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    fake_file = MagicMock()
    fake_file.name = "extract.pdf"
    fake_file.getvalue.return_value = b"extract"
    fake_file.type = "application/pdf"
    patch_streamlit.file_uploader.side_effect = [[], fake_file]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Run Extraction" in a else False
    patch_streamlit.chat_input.return_value = None
    extract_response = MagicMock()
    extract_response.status_code = 400
    extract_response.text = "Bad Extraction"
    patch_requests.post.return_value = extract_response
    import_app()
    patch_streamlit.error.assert_any_call("Error: Bad Extraction")

def test_data_extraction_request_exception(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value = MagicMock(status_code=200)
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    fake_file = MagicMock()
    fake_file.name = "extract.pdf"
    fake_file.getvalue.return_value = b"extract"
    fake_file.type = "application/pdf"
    patch_streamlit.file_uploader.side_effect = [[], fake_file]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Run Extraction" in a else False
    patch_streamlit.chat_input.return_value = None
    patch_requests.post.side_effect = Exception("Extraction failed")
    import_app()
    patch_streamlit.error.assert_any_call("Request failed: Extraction failed")

def test_data_extraction_missing_fields(patch_streamlit, patch_requests, patch_os):
    patch_requests.get.return_value.status_code = 200
    patch_streamlit.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    fake_file = MagicMock()
    fake_file.name = "extract.pdf"
    fake_file.getvalue.return_value = b"extract"
    fake_file.type = "application/pdf"
    patch_streamlit.file_uploader.side_effect = [[], fake_file]
    patch_streamlit.button.side_effect = lambda *a, **kw: True if "Run Extraction" in a else False
    patch_streamlit.chat_input.return_value = None
    extract_response = MagicMock()
    extract_response.status_code = 200
    extract_response.json.return_value = {
        "data": {
            # All fields missing
        }
    }
    patch_requests.post.return_value = extract_response
    import_app()
    patch_streamlit.success.assert_any_call("Extraction complete!")
    # Should display N/A for missing fields
    patch_streamlit.write.assert_any_call("**Reference ID:** N/A")
    patch_streamlit.write.assert_any_call("**Load ID:** N/A")
    patch_streamlit.write.assert_any_call("**Shipper:** N/A")
    patch_streamlit.write.assert_any_call("**Consignee:** N/A")
    patch_streamlit.write.assert_any_call("**Carrier:** N/A")
    patch_streamlit.write.assert_any_call("**MC Number:** N/A")
    patch_streamlit.write.assert_any_call("**Driver:** N/A")
    patch_streamlit.write.assert_any_call("**Truck:** N/A")
    patch_streamlit.write.assert_any_call("**Pickup:** N/A, N/A")
    patch_streamlit.write.assert_any_call("**Pickup Date:** N/A")
    patch_streamlit.write.assert_any_call("**Drop:** N/A, N/A")
    patch_streamlit.write.assert_any_call("**Delivery Date:** N/A")
    patch_streamlit.write.assert_any_call("**Total Rate:** N/A ")
    patch_streamlit.write.assert_any_call("**Equipment:** N/A")
