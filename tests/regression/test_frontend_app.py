import pytest
import builtins
import sys
import types
from unittest.mock import patch, MagicMock, call

# Patch streamlit and requests globally for all tests
@pytest.fixture(autouse=True)
def patch_streamlit_and_requests(monkeypatch):
    # Mock streamlit API
    st_mock = MagicMock()
    # st.session_state is a dict-like object
    st_mock.session_state = {}
    # st.sidebar context manager
    st_mock.sidebar.__enter__.return_value = st_mock.sidebar
    st_mock.sidebar.__exit__.return_value = None
    # st.tabs returns a list of MagicMock context managers
    st_mock.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    # st.file_uploader returns None by default
    st_mock.file_uploader.return_value = None
    # st.button returns False by default
    st_mock.button.return_value = False
    # st.chat_input returns None by default
    st_mock.chat_input.return_value = None
    # st.stop raises RuntimeError to simulate Streamlit's stop
    st_mock.stop.side_effect = RuntimeError("Streamlit stopped execution")
    # st.rerun raises RuntimeError to simulate rerun
    st_mock.rerun.side_effect = RuntimeError("Streamlit rerun")
    # st.spinner context manager
    st_mock.spinner.__enter__.return_value = None
    st_mock.spinner.__exit__.return_value = None
    # st.expander context manager
    st_mock.expander.__enter__.return_value = None
    st_mock.expander.__exit__.return_value = None
    # st.columns returns two MagicMocks
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    # st.chat_message context manager
    st_mock.chat_message.__enter__.return_value = None
    st_mock.chat_message.__exit__.return_value = None

    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    # Patch requests
    requests_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "requests", requests_mock)
    # Patch os
    os_mock = MagicMock()
    os_mock.getenv.side_effect = lambda key, default=None: default
    monkeypatch.setitem(sys.modules, "os", os_mock)
    # Patch json
    import json
    monkeypatch.setitem(sys.modules, "json", json)
    yield

def import_app():
    # Remove frontend.app from sys.modules to force reload
    sys.modules.pop("frontend.app", None)
    import frontend.app

def make_file_mock(name="test.pdf", content=b"abc", type_="application/pdf"):
    file_mock = MagicMock()
    file_mock.name = name
    file_mock.getvalue.return_value = content
    file_mock.type = type_
    return file_mock

def set_streamlit_mock(monkeypatch, **kwargs):
    st = sys.modules["streamlit"]
    for k, v in kwargs.items():
        setattr(st, k, v)

def set_requests_mock(monkeypatch, get=None, post=None):
    requests = sys.modules["requests"]
    if get is not None:
        requests.get.side_effect = get
    if post is not None:
        requests.post.side_effect = post

def set_os_env(monkeypatch, env_dict):
    os_mod = sys.modules["os"]
    os_mod.getenv.side_effect = lambda key, default=None: env_dict.get(key, default)

def get_st_mock():
    return sys.modules["streamlit"]

def get_requests_mock():
    return sys.modules["requests"]

def get_os_mock():
    return sys.modules["os"]

# --- Tests ---

def test_backend_health_check_success(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get)
    # Should not call st.error or st.stop
    st = get_st_mock()
    import_app()
    assert not st.error.called
    assert not st.stop.called

def test_backend_health_check_failure(monkeypatch):
    # Simulate backend /ping returns 500
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 500
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get)
    st = get_st_mock()
    # st.button returns False (no retry)
    st.button.return_value = False
    with pytest.raises(RuntimeError, match="Streamlit stopped execution"):
        import_app()
    st.error.assert_any_call("⚠️ Cannot connect to backend at http://localhost:8000. Please ensure the backend is running.")

def test_backend_health_check_retry(monkeypatch):
    # Simulate backend /ping returns 500
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 500
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get)
    st = get_st_mock()
    # st.button returns True (retry)
    st.button.return_value = True
    with pytest.raises(RuntimeError, match="Streamlit rerun"):
        import_app()
    st.rerun.assert_called_once()

def test_backend_health_check_exception(monkeypatch):
    # Simulate requests.get raises exception
    def fake_get(url, *a, **k):
        raise Exception("network error")
    set_requests_mock(monkeypatch, get=fake_get)
    st = get_st_mock()
    st.button.return_value = False
    with pytest.raises(RuntimeError, match="Streamlit stopped execution"):
        import_app()
    st.error.assert_any_call("⚠️ Cannot connect to backend at http://localhost:8000. Please ensure the backend is running.")

def test_upload_and_index_happy_path(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /upload returns 200 with message and extractions
    def fake_post(url, files=None, *a, **k):
        class Resp:
            status_code = 200
            def json(self):
                return {
                    "message": "Indexed successfully!",
                    "extractions": [
                        {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
                        {"filename": "doc2.pdf", "structured_data_extracted": False, "text_chunks": 3}
                    ]
                }
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    # Simulate file upload and button click
    file1 = make_file_mock("doc1.pdf")
    file2 = make_file_mock("doc2.pdf")
    st.file_uploader.side_effect = [ [file1, file2], None, None ]
    st.button.side_effect = [True, False, False]
    import_app()
    st.success.assert_any_call("Indexed successfully!")
    # Extraction summary should be shown
    assert st.write.call_args_list
    found_doc1 = any("**doc1.pdf**" in str(args) for args, _ in st.write.call_args_list)
    found_doc2 = any("**doc2.pdf**" in str(args) for args, _ in st.write.call_args_list)
    assert found_doc1 and found_doc2

def test_upload_and_index_no_files(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get)
    st = get_st_mock()
    st.file_uploader.side_effect = [ [], None, None ]
    st.button.side_effect = [True, False, False]
    import_app()
    st.warning.assert_any_call("Please upload at least one document.")

def test_upload_and_index_backend_error(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /upload returns 400
    def fake_post(url, files=None, *a, **k):
        class Resp:
            status_code = 400
            text = "Bad request"
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    file1 = make_file_mock("doc1.pdf")
    st.file_uploader.side_effect = [ [file1], None, None ]
    st.button.side_effect = [True, False, False]
    import_app()
    st.error.assert_any_call("Error: Bad request")

def test_upload_and_index_request_exception(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /upload raises exception
    def fake_post(url, files=None, *a, **k):
        raise Exception("upload failed")
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    file1 = make_file_mock("doc1.pdf")
    st.file_uploader.side_effect = [ [file1], None, None ]
    st.button.side_effect = [True, False, False]
    import_app()
    st.error.assert_any_call("Request failed: upload failed")

def test_chat_qa_happy_path(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /ask returns 200 with answer, confidence, sources
    def fake_post(url, json=None, *a, **k):
        class Resp:
            status_code = 200
            def json(self):
                return {
                    "answer": "The agreed rate is $1000.",
                    "confidence_score": 0.95,
                    "sources": [
                        {"metadata": {"source": "doc1.pdf"}, "text": "Relevant text"}
                    ]
                }
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    # Simulate chat input
    st.chat_input.side_effect = [ "What is the agreed rate?", None, None ]
    st.session_state.messages = []
    import_app()
    # Should display answer and confidence
    st.markdown.assert_any_call("The agreed rate is $1000.")
    st.caption.assert_any_call("Confidence Score: 0.95")
    # Should save assistant message to session_state
    assert any(
        m.get("role") == "assistant" and m.get("content") == "The agreed rate is $1000."
        for m in st.session_state.messages
    )

def test_chat_qa_backend_error(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /ask returns 500
    def fake_post(url, json=None, *a, **k):
        class Resp:
            status_code = 500
            text = "Internal error"
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    st.chat_input.side_effect = [ "What is the agreed rate?", None, None ]
    st.session_state.messages = []
    import_app()
    st.error.assert_any_call("Error: Internal error")

def test_chat_qa_request_exception(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /ask raises exception
    def fake_post(url, json=None, *a, **k):
        raise Exception("ask failed")
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    st.chat_input.side_effect = [ "What is the agreed rate?", None, None ]
    st.session_state.messages = []
    import_app()
    st.error.assert_any_call("Request failed: ask failed")

def test_data_extraction_happy_path(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /extract returns 200 with data
    def fake_post(url, files=None, *a, **k):
        class Resp:
            status_code = 200
            def json(self):
                return {
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
                        "rate_info": {"total_rate": "1000", "currency": "USD"},
                        "equipment_type": "Van"
                    }
                }
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    file1 = make_file_mock("extract.pdf")
    # The third tab's uploader uses key="extraction_uploader"
    st.file_uploader.side_effect = [ None, None, file1 ]
    st.button.side_effect = [False, False, True]
    import_app()
    st.success.assert_any_call("Extraction complete!")
    # Should display some fields
    assert st.write.call_args_list
    found_ref = any("**Reference ID:** REF123" in str(args) for args, _ in st.write.call_args_list)
    found_carrier = any("**Carrier:** CarrierX" in str(args) for args, _ in st.write.call_args_list)
    found_rate = any("**Total Rate:** 1000 USD" in str(args) for args, _ in st.write.call_args_list)
    assert found_ref and found_carrier and found_rate

def test_data_extraction_no_file(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get)
    st = get_st_mock()
    st.file_uploader.side_effect = [ None, None, None ]
    st.button.side_effect = [False, False, True]
    import_app()
    st.warning.assert_any_call("Please upload a document.")

def test_data_extraction_backend_error(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /extract returns 400
    def fake_post(url, files=None, *a, **k):
        class Resp:
            status_code = 400
            text = "Bad extract"
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    file1 = make_file_mock("extract.pdf")
    st.file_uploader.side_effect = [ None, None, file1 ]
    st.button.side_effect = [False, False, True]
    import_app()
    st.error.assert_any_call("Error: Bad extract")

def test_data_extraction_request_exception(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /extract raises exception
    def fake_post(url, files=None, *a, **k):
        raise Exception("extract failed")
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    file1 = make_file_mock("extract.pdf")
    st.file_uploader.side_effect = [ None, None, file1 ]
    st.button.side_effect = [False, False, True]
    import_app()
    st.error.assert_any_call("Request failed: extract failed")

def test_upload_and_index_with_errors(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /upload returns 200 with errors
    def fake_post(url, files=None, *a, **k):
        class Resp:
            status_code = 200
            def json(self):
                return {
                    "message": "Indexed with warnings.",
                    "extractions": [],
                    "errors": ["File doc3.pdf failed to process."]
                }
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    file1 = make_file_mock("doc3.pdf")
    st.file_uploader.side_effect = [ [file1], None, None ]
    st.button.side_effect = [True, False, False]
    import_app()
    st.warning.assert_any_call("File doc3.pdf failed to process.")

def test_data_extraction_missing_fields(monkeypatch):
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        class Resp:
            status_code = 200
        return Resp()
    # Simulate /extract returns 200 with missing fields
    def fake_post(url, files=None, *a, **k):
        class Resp:
            status_code = 200
            def json(self):
                return {
                    "data": {
                        # Only a few fields
                        "reference_id": None,
                        "carrier": None,
                        "driver": None,
                        "pickup": None,
                        "drop": None,
                        "rate_info": None
                    }
                }
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get, post=fake_post)
    st = get_st_mock()
    file1 = make_file_mock("extract.pdf")
    st.file_uploader.side_effect = [ None, None, file1 ]
    st.button.side_effect = [False, False, True]
    import_app()
    # Should display N/A for missing fields
    found_na = any("N/A" in str(args) for args, _ in st.write.call_args_list)
    assert found_na

def test_sidebar_backend_url_env(monkeypatch):
    # Simulate BACKEND_URL env var
    set_os_env(monkeypatch, {"BACKEND_URL": "http://env-backend:9000"})
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        assert url.startswith("http://env-backend:9000")
        class Resp:
            status_code = 200
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get)
    st = get_st_mock()
    import_app()
    st.text_input.assert_any_call(
        "Backend API URL",
        value="http://env-backend:9000",
        help="The URL of the FastAPI backend."
    )

def test_sidebar_backend_url_default(monkeypatch):
    # No BACKEND_URL env var
    set_os_env(monkeypatch, {})
    # Simulate backend /ping returns 200
    def fake_get(url, *a, **k):
        assert url.startswith("http://localhost:8000")
        class Resp:
            status_code = 200
        return Resp()
    set_requests_mock(monkeypatch, get=fake_get)
    st = get_st_mock()
    import_app()
    st.text_input.assert_any_call(
        "Backend API URL",
        value="http://localhost:8000",
        help="The URL of the FastAPI backend."
    )
