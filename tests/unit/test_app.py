import pytest
from unittest.mock import patch, MagicMock, call
import builtins

# Patch streamlit and requests globally for all tests
@pytest.fixture(autouse=True)
def patch_streamlit_and_requests(monkeypatch):
    # Patch streamlit
    st_mock = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, 'streamlit', st_mock)
    # Patch requests
    requests_mock = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, 'requests', requests_mock)
    # Patch os
    os_mock = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, 'os', os_mock)
    yield

# Helper to import the app after patching
def import_app():
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        importlib.reload(sys.modules["frontend.app"])
    else:
        import frontend.app

def make_file_mock(name="doc.pdf", content=b"abc", type_="application/pdf"):
    file_mock = MagicMock()
    file_mock.name = name
    file_mock.getvalue.return_value = content
    file_mock.type = type_
    return file_mock

def make_response_mock(status_code=200, json_data=None, text="OK"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp

def test_check_backend_happy(monkeypatch):
    # Simulate backend up
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    # Should not call st.error or st.stop
    st.error.reset_mock()
    st.stop.reset_mock()
    import_app()
    assert not st.error.called
    assert not st.stop.called

def test_check_backend_down(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(500)
    st.button.return_value = False
    st.error.reset_mock()
    st.stop.reset_mock()
    import_app()
    st.error.assert_called()
    st.stop.assert_called()

def test_check_backend_down_and_retry(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(500)
    st.button.return_value = True
    st.rerun.reset_mock()
    import_app()
    st.rerun.assert_called()

def test_check_backend_exception(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.side_effect = Exception("fail")
    st.button.return_value = False
    import_app()
    st.error.assert_called()
    st.stop.assert_called()

def test_upload_and_index_happy(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    # Backend health up
    requests.get.return_value = make_response_mock(200)
    # Simulate file upload
    file1 = make_file_mock("a.pdf", b"abc", "application/pdf")
    file2 = make_file_mock("b.docx", b"def", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    st.file_uploader.side_effect = [[file1, file2], None, None]
    st.button.side_effect = [False, True, False]
    # Simulate upload response
    result = {
        "message": "Indexed!",
        "extractions": [
            {"filename": "a.pdf", "structured_data_extracted": True, "text_chunks": 3},
            {"filename": "b.docx", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": ["Minor error"]
    }
    requests.post.return_value = make_response_mock(200, result)
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    st.expander.return_value.__enter__.return_value = MagicMock()
    st.expander.return_value.__exit__.return_value = None
    import_app()
    st.success.assert_any_call("Indexed!")
    st.warning.assert_any_call("Minor error")

def test_upload_and_index_no_files(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], None, None]
    st.button.side_effect = [False, True, False]
    import_app()
    st.warning.assert_any_call("Please upload at least one document.")

def test_upload_and_index_backend_error(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    file1 = make_file_mock("a.pdf", b"abc", "application/pdf")
    st.file_uploader.side_effect = [[file1], None, None]
    st.button.side_effect = [False, True, False]
    requests.post.return_value = make_response_mock(500, None, "Internal Error")
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    import_app()
    st.error.assert_any_call("Error: Internal Error")

def test_upload_and_index_request_exception(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    file1 = make_file_mock("a.pdf", b"abc", "application/pdf")
    st.file_uploader.side_effect = [[file1], None, None]
    st.button.side_effect = [False, True, False]
    requests.post.side_effect = Exception("fail upload")
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    import_app()
    st.error.assert_any_call("Request failed: fail upload")

def test_chat_qa_happy(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], None, None]
    st.button.side_effect = [False, False, False]
    # Simulate chat input
    st.session_state = {}
    st.session_state["messages"] = []
    st.chat_input.return_value = "What is the rate?"
    # Simulate backend response
    answer = "The rate is $1000."
    confidence = 0.98
    sources = [{"metadata": {"source": "doc1"}, "text": "Relevant text"}]
    data = {"answer": answer, "confidence_score": confidence, "sources": sources}
    requests.post.return_value = make_response_mock(200, data)
    st.chat_message.return_value.__enter__.return_value = MagicMock()
    st.chat_message.return_value.__exit__.return_value = None
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    st.expander.return_value.__enter__.return_value = MagicMock()
    st.expander.return_value.__exit__.return_value = None
    import_app()
    st.markdown.assert_any_call(answer)
    st.caption.assert_any_call(f"Confidence Score: {confidence:.2f}")

def test_chat_qa_backend_error(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], None, None]
    st.button.side_effect = [False, False, False]
    st.session_state = {}
    st.session_state["messages"] = []
    st.chat_input.return_value = "What is the rate?"
    requests.post.return_value = make_response_mock(500, None, "Backend error")
    st.chat_message.return_value.__enter__.return_value = MagicMock()
    st.chat_message.return_value.__exit__.return_value = None
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    import_app()
    st.error.assert_any_call("Error: Backend error")

def test_chat_qa_request_exception(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], None, None]
    st.button.side_effect = [False, False, False]
    st.session_state = {}
    st.session_state["messages"] = []
    st.chat_input.return_value = "What is the rate?"
    requests.post.side_effect = Exception("fail ask")
    st.chat_message.return_value.__enter__.return_value = MagicMock()
    st.chat_message.return_value.__exit__.return_value = None
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    import_app()
    st.error.assert_any_call("Request failed: fail ask")

def test_data_extraction_happy(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], make_file_mock("extract.pdf", b"abc", "application/pdf"), None]
    st.button.side_effect = [False, False, True]
    data = {
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
            "rate_info": {"total_rate": "1000", "currency": "USD"},
            "equipment_type": "Van"
        }
    }
    requests.post.return_value = make_response_mock(200, data)
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    st.columns.return_value = (MagicMock(), MagicMock())
    st.expander.return_value.__enter__.return_value = MagicMock()
    st.expander.return_value.__exit__.return_value = None
    import_app()
    st.success.assert_any_call("Extraction complete!")
    st.json.assert_any_call(data["data"])

def test_data_extraction_backend_error(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], make_file_mock("extract.pdf", b"abc", "application/pdf"), None]
    st.button.side_effect = [False, False, True]
    requests.post.return_value = make_response_mock(500, None, "Extraction error")
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    st.columns.return_value = (MagicMock(), MagicMock())
    import_app()
    st.error.assert_any_call("Error: Extraction error")

def test_data_extraction_request_exception(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], make_file_mock("extract.pdf", b"abc", "application/pdf"), None]
    st.button.side_effect = [False, False, True]
    requests.post.side_effect = Exception("fail extract")
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    st.columns.return_value = (MagicMock(), MagicMock())
    import_app()
    st.error.assert_any_call("Request failed: fail extract")

def test_data_extraction_no_file(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], None, None]
    st.button.side_effect = [False, False, True]
    import_app()
    st.warning.assert_any_call("Please upload a document.")

def test_data_extraction_missing_fields(monkeypatch):
    import sys
    st = sys.modules['streamlit']
    requests = sys.modules['requests']
    os = sys.modules['os']
    os.getenv.return_value = "http://backend"
    st.text_input.return_value = "http://backend"
    requests.get.return_value = make_response_mock(200)
    st.file_uploader.side_effect = [[], make_file_mock("extract.pdf", b"abc", "application/pdf"), None]
    st.button.side_effect = [False, False, True]
    # Data with missing nested fields
    data = {
        "data": {
            "reference_id": None,
            "load_id": None,
            "shipper": None,
            "consignee": None,
            "carrier": None,
            "driver": None,
            "pickup": None,
            "shipping_date": None,
            "drop": None,
            "delivery_date": None,
            "rate_info": None,
            "equipment_type": None
        }
    }
    requests.post.return_value = make_response_mock(200, data)
    st.spinner.return_value.__enter__.return_value = None
    st.spinner.return_value.__exit__.return_value = None
    st.columns.return_value = (MagicMock(), MagicMock())
    st.expander.return_value.__enter__.return_value = MagicMock()
    st.expander.return_value.__exit__.return_value = None
    import_app()
    st.success.assert_any_call("Extraction complete!")
    st.json.assert_any_call(data["data"])
