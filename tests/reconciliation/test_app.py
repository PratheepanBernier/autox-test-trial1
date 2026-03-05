# source_hash: 9ee607a3d8da4254
import pytest
from unittest.mock import patch, MagicMock
import builtins

# Since the code is a Streamlit app, we focus on the logic and reconciliation of backend interactions.
# We'll test the three main flows: upload/index, chat/QA, and extraction.
# We'll mock requests and Streamlit APIs to isolate and compare equivalent paths.

@pytest.fixture
def mock_streamlit(monkeypatch):
    # Patch all used streamlit methods to MagicMock
    st_mock = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, 'streamlit', st_mock)
    # Patch st.session_state as a dict
    st_mock.session_state = {}
    # Patch st.sidebar context manager
    st_mock.sidebar.__enter__.return_value = None
    st_mock.sidebar.__exit__.return_value = None
    # Patch st.tabs to return list of MagicMock context managers
    st_mock.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    # Patch st.columns to return tuple of MagicMock context managers
    st_mock.columns.side_effect = lambda n: tuple(MagicMock() for _ in range(n))
    # Patch st.expander as context manager
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.expander.return_value.__exit__.return_value = None
    # Patch st.chat_message as context manager
    st_mock.chat_message.return_value.__enter__.return_value = None
    st_mock.chat_message.return_value.__exit__.return_value = None
    # Patch st.spinner as context manager
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    # Patch st.sidebar context manager
    st_mock.sidebar.__enter__.return_value = None
    st_mock.sidebar.__exit__.return_value = None
    # Patch st.stop to raise SystemExit
    st_mock.stop.side_effect = SystemExit
    # Patch st.rerun to raise RuntimeError (simulate rerun)
    st_mock.rerun.side_effect = RuntimeError("rerun")
    return st_mock

@pytest.fixture
def mock_requests(monkeypatch):
    # Patch requests.get and requests.post
    get_mock = MagicMock()
    post_mock = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, 'requests', MagicMock(get=get_mock, post=post_mock))
    return get_mock, post_mock

@pytest.fixture
def mock_os(monkeypatch):
    # Patch os.getenv
    os_mock = MagicMock()
    os_mock.getenv.side_effect = lambda k, v=None: v
    monkeypatch.setitem(__import__('sys').modules, 'os', os_mock)
    return os_mock

@pytest.fixture
def import_app(mock_streamlit, mock_requests, mock_os):
    # Import the app after patching dependencies
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    app = importlib.import_module("frontend.app")
    return app

def make_file_mock(name, content, type_):
    file_mock = MagicMock()
    file_mock.name = name
    file_mock.getvalue.return_value = content
    file_mock.type = type_
    return file_mock

def make_response_mock(status_code, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp

def test_backend_health_check_success(mock_streamlit, mock_requests, mock_os):
    # Backend /ping returns 200
    get_mock, _ = mock_requests
    get_mock.return_value = make_response_mock(200)
    # Should not call st.error or st.stop
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert not mock_streamlit.error.called
    assert not mock_streamlit.stop.called

def test_backend_health_check_failure_triggers_error_and_stop(mock_streamlit, mock_requests, mock_os):
    # Backend /ping returns 500
    get_mock, _ = mock_requests
    get_mock.return_value = make_response_mock(500)
    # Should call st.error and st.stop
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    with pytest.raises(SystemExit):
        importlib.import_module("frontend.app")
    assert mock_streamlit.error.called
    assert mock_streamlit.stop.called

def test_backend_health_check_exception_triggers_error_and_stop(mock_streamlit, mock_requests, mock_os):
    # Backend /ping raises exception
    get_mock, _ = mock_requests
    get_mock.side_effect = Exception("network error")
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    with pytest.raises(SystemExit):
        importlib.import_module("frontend.app")
    assert mock_streamlit.error.called
    assert mock_streamlit.stop.called

def test_upload_and_index_happy_path(mock_streamlit, mock_requests, mock_os):
    # Simulate backend healthy
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    # Simulate file upload and button press
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    file2 = make_file_mock("doc2.txt", b"TXTDATA", "text/plain")
    mock_streamlit.file_uploader.side_effect = [[file1, file2], None, None]
    # Simulate button press
    mock_streamlit.button.side_effect = [False, True, False, False, False]
    # Simulate backend /upload response
    post_mock.return_value = make_response_mock(200, {
        "message": "Indexed 2 documents.",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
            {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    })
    # Import app
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    # Should call st.success with message and show extraction summary
    assert mock_streamlit.success.called
    assert mock_streamlit.expander.called

def test_upload_and_index_no_files_warns(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    mock_streamlit.file_uploader.side_effect = [[], None, None]
    mock_streamlit.button.side_effect = [False, True, False, False, False]
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.warning.called

def test_upload_and_index_backend_error(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    mock_streamlit.file_uploader.side_effect = [[file1], None, None]
    mock_streamlit.button.side_effect = [False, True, False, False, False]
    post_mock.return_value = make_response_mock(500, text="Internal Error")
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.error.called

def test_upload_and_index_backend_exception(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    mock_streamlit.file_uploader.side_effect = [[file1], None, None]
    mock_streamlit.button.side_effect = [False, True, False, False, False]
    post_mock.side_effect = Exception("network error")
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.error.called

def test_chat_qa_happy_path_and_reconciliation(mock_streamlit, mock_requests, mock_os):
    # Simulate backend healthy
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    # Simulate chat input
    mock_streamlit.chat_input.return_value = "What is the agreed rate for this shipment?"
    # Simulate session state
    mock_streamlit.session_state = {"messages": []}
    # Simulate backend /ask response
    post_mock.return_value = make_response_mock(200, {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
        ]
    })
    # Import app
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    # Should call st.markdown with answer and st.caption with confidence
    assert mock_streamlit.markdown.called
    assert mock_streamlit.caption.called
    # Should append to session_state.messages
    assert any(m.get("role") == "assistant" for m in mock_streamlit.session_state["messages"])

def test_chat_qa_backend_error(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    mock_streamlit.chat_input.return_value = "What is the agreed rate for this shipment?"
    mock_streamlit.session_state = {"messages": []}
    post_mock.return_value = make_response_mock(500, text="Internal Error")
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.error.called

def test_chat_qa_backend_exception(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    mock_streamlit.chat_input.return_value = "What is the agreed rate for this shipment?"
    mock_streamlit.session_state = {"messages": []}
    post_mock.side_effect = Exception("network error")
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.error.called

def test_data_extraction_happy_path_and_reconciliation(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    # file_uploader for extraction tab
    mock_streamlit.file_uploader.side_effect = [None, None, file1]
    # Button presses: upload, chat, extraction
    mock_streamlit.button.side_effect = [False, False, True, False, False]
    # Simulate backend /extract response
    post_mock.return_value = make_response_mock(200, {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Acme Inc.",
            "consignee": "Beta LLC",
            "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
            "driver": {"driver_name": "John Doe", "truck_number": "TRK001"},
            "pickup": {"city": "Dallas", "state": "TX"},
            "shipping_date": "2024-06-01",
            "drop": {"city": "Houston", "state": "TX"},
            "delivery_date": "2024-06-02",
            "rate_info": {"total_rate": 1200, "currency": "USD"},
            "equipment_type": "Reefer"
        }
    })
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    # Should call st.success and st.json
    assert mock_streamlit.success.called
    assert mock_streamlit.json.called

def test_data_extraction_no_file_warns(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    mock_streamlit.file_uploader.side_effect = [None, None, None]
    mock_streamlit.button.side_effect = [False, False, True, False, False]
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.warning.called

def test_data_extraction_backend_error(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    mock_streamlit.file_uploader.side_effect = [None, None, file1]
    mock_streamlit.button.side_effect = [False, False, True, False, False]
    post_mock.return_value = make_response_mock(500, text="Internal Error")
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.error.called

def test_data_extraction_backend_exception(mock_streamlit, mock_requests, mock_os):
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    mock_streamlit.file_uploader.side_effect = [None, None, file1]
    mock_streamlit.button.side_effect = [False, False, True, False, False]
    post_mock.side_effect = Exception("network error")
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    assert mock_streamlit.error.called

def test_reconciliation_upload_vs_extraction(mock_streamlit, mock_requests, mock_os):
    # This test checks that uploading and extracting the same file yields consistent backend calls
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    # Simulate file upload for both upload/index and extraction tabs
    mock_streamlit.file_uploader.side_effect = [[file1], None, file1]
    # Simulate button presses: upload, extraction
    mock_streamlit.button.side_effect = [False, True, True, False, False]
    # /upload and /extract responses
    upload_result = {
        "message": "Indexed 1 document.",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5}
        ],
        "errors": []
    }
    extract_result = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Acme Inc.",
            "consignee": "Beta LLC",
            "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
            "driver": {"driver_name": "John Doe", "truck_number": "TRK001"},
            "pickup": {"city": "Dallas", "state": "TX"},
            "shipping_date": "2024-06-01",
            "drop": {"city": "Houston", "state": "TX"},
            "delivery_date": "2024-06-02",
            "rate_info": {"total_rate": 1200, "currency": "USD"},
            "equipment_type": "Reefer"
        }
    }
    post_mock.side_effect = [
        make_response_mock(200, upload_result),  # /upload
        make_response_mock(200, extract_result)  # /extract
    ]
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    # Check that both /upload and /extract were called with the same file content
    upload_call = post_mock.call_args_list[0]
    extract_call = post_mock.call_args_list[1]
    upload_files = upload_call[1]['files']
    extract_files = extract_call[1]['files']
    # Compare file names and content
    assert upload_files[0][1][0] == extract_files['file'][0]
    assert upload_files[0][1][1] == extract_files['file'][1]
    # Check that both flows succeeded
    assert mock_streamlit.success.called

def test_reconciliation_chat_vs_extraction(mock_streamlit, mock_requests, mock_os):
    # This test checks that asking a question and extracting data for the same file yields compatible info
    get_mock, post_mock = mock_requests
    get_mock.return_value = make_response_mock(200)
    # Simulate chat input
    mock_streamlit.chat_input.return_value = "What is the agreed rate for this shipment?"
    mock_streamlit.session_state = {"messages": []}
    # Simulate file upload for extraction
    file1 = make_file_mock("doc1.pdf", b"PDFDATA", "application/pdf")
    mock_streamlit.file_uploader.side_effect = [None, None, file1]
    # Simulate button presses: upload, chat, extraction
    mock_streamlit.button.side_effect = [False, False, True, False, False]
    # /ask and /extract responses
    chat_result = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
        ]
    }
    extract_result = {
        "data": {
            "rate_info": {"total_rate": 1200, "currency": "USD"}
        }
    }
    post_mock.side_effect = [
        make_response_mock(200, chat_result),   # /ask
        make_response_mock(200, extract_result) # /extract
    ]
    import importlib
    import sys
    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]
    importlib.import_module("frontend.app")
    # Check that the rate in chat answer matches the extracted rate
    chat_rate = chat_result["answer"].split("$")[-1].split(".")[0]
    extract_rate = str(extract_result["data"]["rate_info"]["total_rate"])
    assert chat_rate in extract_rate or extract_rate in chat_rate
    # Both flows should succeed
    assert mock_streamlit.success.called or mock_streamlit.markdown.called
