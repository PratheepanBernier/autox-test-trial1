import builtins
import io
import os
import sys
import types
import pytest

import frontend.app as app

import streamlit as st
import requests

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit UI functions to no-op or record calls
    # so the app can be imported and run without UI
    dummy = MagicMock()
    monkeypatch.setattr(st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(st, "title", lambda *a, **k: None)
    monkeypatch.setattr(st, "header", lambda *a, **k: None)
    monkeypatch.setattr(st, "write", lambda *a, **k: None)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "divider", lambda *a, **k: None)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "json", lambda *a, **k: None)
    monkeypatch.setattr(st, "spinner", lambda *a, **k: dummy.__enter__)
    monkeypatch.setattr(st, "expander", lambda *a, **k: dummy.__enter__)
    monkeypatch.setattr(st, "columns", lambda *a, **k: (dummy, dummy))
    monkeypatch.setattr(st, "tabs", lambda labels: [dummy, dummy, dummy])
    monkeypatch.setattr(st, "sidebar", dummy.__enter__)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: None)
    monkeypatch.setattr(st, "button", lambda *a, **k: False)
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(st, "rerun", lambda: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(st, "chat_message", dummy.__enter__)
    monkeypatch.setattr(st, "chat_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "session_state", {})
    yield

@pytest.fixture
def mock_requests(monkeypatch):
    # Patch requests.get and requests.post
    get_mock = MagicMock()
    post_mock = MagicMock()
    monkeypatch.setattr(requests, "get", get_mock)
    monkeypatch.setattr(requests, "post", post_mock)
    return get_mock, post_mock

def make_response(status_code=200, json_data=None, text="OK"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = Exception("No JSON")
    return resp

def test_check_backend_success(monkeypatch, mock_requests):
    get_mock, _ = mock_requests
    get_mock.return_value = make_response(200)
    # Should return True if backend responds 200
    assert app.check_backend() is True

def test_check_backend_failure_status(monkeypatch, mock_requests):
    get_mock, _ = mock_requests
    get_mock.return_value = make_response(500)
    assert app.check_backend() is False

def test_check_backend_exception(monkeypatch, mock_requests):
    get_mock, _ = mock_requests
    get_mock.side_effect = Exception("Connection error")
    assert app.check_backend() is False

def test_upload_and_index_happy_path(monkeypatch, mock_requests):
    # Simulate file upload and successful backend response
    _, post_mock = mock_requests
    files = [
        MagicMock(name="file1", spec=["name", "getvalue", "type"]),
        MagicMock(name="file2", spec=["name", "getvalue", "type"])
    ]
    files[0].name = "doc1.pdf"
    files[0].getvalue.return_value = b"PDFDATA"
    files[0].type = "application/pdf"
    files[1].name = "doc2.txt"
    files[1].getvalue.return_value = b"TXTDATA"
    files[1].type = "text/plain"
    # Patch file_uploader to return files
    with patch.object(st, "file_uploader", return_value=files):
        # Patch button to simulate click
        with patch.object(st, "button", side_effect=lambda label: label == "Process & Index Documents"):
            # Patch spinner context
            with patch.object(st, "spinner", return_value=(lambda: (yield))()):
                # Patch requests.post to return success
                post_mock.return_value = make_response(200, {
                    "message": "Indexed 2 documents.",
                    "extractions": [
                        {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
                        {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
                    ],
                    "errors": []
                })
                # Run the relevant code block
                # Simulate the tab context
                with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                    # Import app again to re-run the code
                    import importlib
                    importlib.reload(app)
                # Check that post was called with correct files
                assert post_mock.called
                args, kwargs = post_mock.call_args
                assert args[0].endswith("/upload")
                assert "files" in kwargs

def test_upload_and_index_no_files(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    # Patch file_uploader to return None
    with patch.object(st, "file_uploader", return_value=None):
        with patch.object(st, "button", side_effect=lambda label: label == "Process & Index Documents"):
            with patch.object(st, "warning") as warn_mock:
                with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                    import importlib
                    importlib.reload(app)
                warn_mock.assert_any_call("Please upload at least one document.")

def test_upload_and_index_backend_error(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    files = [
        MagicMock(name="file1", spec=["name", "getvalue", "type"])
    ]
    files[0].name = "doc1.pdf"
    files[0].getvalue.return_value = b"PDFDATA"
    files[0].type = "application/pdf"
    with patch.object(st, "file_uploader", return_value=files):
        with patch.object(st, "button", side_effect=lambda label: label == "Process & Index Documents"):
            with patch.object(st, "spinner", return_value=(lambda: (yield))()):
                post_mock.return_value = make_response(400, text="Bad Request")
                with patch.object(st, "error") as err_mock:
                    with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                        import importlib
                        importlib.reload(app)
                    err_mock.assert_any_call("Error: Bad Request")

def test_upload_and_index_request_exception(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    files = [
        MagicMock(name="file1", spec=["name", "getvalue", "type"])
    ]
    files[0].name = "doc1.pdf"
    files[0].getvalue.return_value = b"PDFDATA"
    files[0].type = "application/pdf"
    with patch.object(st, "file_uploader", return_value=files):
        with patch.object(st, "button", side_effect=lambda label: label == "Process & Index Documents"):
            with patch.object(st, "spinner", return_value=(lambda: (yield))()):
                post_mock.side_effect = Exception("Network error")
                with patch.object(st, "error") as err_mock:
                    with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                        import importlib
                        importlib.reload(app)
                    assert any("Request failed: Network error" in str(call.args[0]) for call in err_mock.mock_calls)

def test_chat_qa_happy_path(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    # Patch chat_input to simulate user prompt
    with patch.object(st, "chat_input", return_value="What is the agreed rate?"):
        # Patch session_state for chat history
        st.session_state.clear()
        # Patch requests.post to return answer
        post_mock.return_value = make_response(200, {
            "answer": "The agreed rate is $1200.",
            "confidence_score": 0.98,
            "sources": [
                {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
            ]
        })
        with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
            import importlib
            importlib.reload(app)
        assert post_mock.called
        args, kwargs = post_mock.call_args
        assert args[0].endswith("/ask")
        assert kwargs["json"]["question"] == "What is the agreed rate?"

def test_chat_qa_backend_error(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    with patch.object(st, "chat_input", return_value="What is the agreed rate?"):
        st.session_state.clear()
        post_mock.return_value = make_response(400, text="Bad Request")
        with patch.object(st, "error") as err_mock:
            with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                import importlib
                importlib.reload(app)
            err_mock.assert_any_call("Error: Bad Request")

def test_chat_qa_request_exception(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    with patch.object(st, "chat_input", return_value="What is the agreed rate?"):
        st.session_state.clear()
        post_mock.side_effect = Exception("Network error")
        with patch.object(st, "error") as err_mock:
            with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                import importlib
                importlib.reload(app)
            assert any("Request failed: Network error" in str(call.args[0]) for call in err_mock.mock_calls)

def test_data_extraction_happy_path(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    file = MagicMock(name="file", spec=["name", "getvalue", "type"])
    file.name = "doc1.pdf"
    file.getvalue.return_value = b"PDFDATA"
    file.type = "application/pdf"
    with patch.object(st, "file_uploader", return_value=file):
        with patch.object(st, "button", side_effect=lambda label: label == "Run Extraction"):
            with patch.object(st, "spinner", return_value=(lambda: (yield))()):
                post_mock.return_value = make_response(200, {
                    "data": {
                        "reference_id": "REF123",
                        "load_id": "LOAD456",
                        "shipper": "Shipper Inc.",
                        "consignee": "Consignee LLC",
                        "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
                        "driver": {"driver_name": "John Doe", "truck_number": "TRK123"},
                        "pickup": {"city": "Dallas", "state": "TX"},
                        "shipping_date": "2024-06-01",
                        "drop": {"city": "Houston", "state": "TX"},
                        "delivery_date": "2024-06-02",
                        "rate_info": {"total_rate": 1200, "currency": "USD"},
                        "equipment_type": "Van"
                    }
                })
                with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                    import importlib
                    importlib.reload(app)
                assert post_mock.called
                args, kwargs = post_mock.call_args
                assert args[0].endswith("/extract")
                assert "files" in kwargs or "file" in kwargs

def test_data_extraction_no_file(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    with patch.object(st, "file_uploader", return_value=None):
        with patch.object(st, "button", side_effect=lambda label: label == "Run Extraction"):
            with patch.object(st, "warning") as warn_mock:
                with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                    import importlib
                    importlib.reload(app)
                warn_mock.assert_any_call("Please upload a document.")

def test_data_extraction_backend_error(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    file = MagicMock(name="file", spec=["name", "getvalue", "type"])
    file.name = "doc1.pdf"
    file.getvalue.return_value = b"PDFDATA"
    file.type = "application/pdf"
    with patch.object(st, "file_uploader", return_value=file):
        with patch.object(st, "button", side_effect=lambda label: label == "Run Extraction"):
            with patch.object(st, "spinner", return_value=(lambda: (yield))()):
                post_mock.return_value = make_response(400, text="Bad Request")
                with patch.object(st, "error") as err_mock:
                    with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                        import importlib
                        importlib.reload(app)
                    err_mock.assert_any_call("Error: Bad Request")

def test_data_extraction_request_exception(monkeypatch, mock_requests):
    _, post_mock = mock_requests
    file = MagicMock(name="file", spec=["name", "getvalue", "type"])
    file.name = "doc1.pdf"
    file.getvalue.return_value = b"PDFDATA"
    file.type = "application/pdf"
    with patch.object(st, "file_uploader", return_value=file):
        with patch.object(st, "button", side_effect=lambda label: label == "Run Extraction"):
            with patch.object(st, "spinner", return_value=(lambda: (yield))()):
                post_mock.side_effect = Exception("Network error")
                with patch.object(st, "error") as err_mock:
                    with patch.object(st, "tabs", return_value=[types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s,a,b,c: None) for _ in range(3)]):
                        import importlib
                        importlib.reload(app)
                    assert any("Request failed: Network error" in str(call.args[0]) for call in err_mock.mock_calls)

def test_backend_unreachable(monkeypatch, mock_requests):
    get_mock, _ = mock_requests
    get_mock.return_value = make_response(500)
    # Patch st.error and st.stop to check error and stop called
    with patch.object(st, "error") as err_mock, patch.object(st, "stop", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            import importlib
            importlib.reload(app)
        assert err_mock.call_count >= 1
        assert any("Cannot connect to backend" in str(call.args[0]) for call in err_mock.mock_calls)

def test_backend_unreachable_retry(monkeypatch, mock_requests):
    get_mock, _ = mock_requests
    get_mock.return_value = make_response(500)
    # Patch st.error, st.button, st.rerun, st.stop
    with patch.object(st, "error") as err_mock, \
         patch.object(st, "button", side_effect=lambda label: label == "Retry Connection"), \
         patch.object(st, "rerun", side_effect=SystemExit), \
         patch.object(st, "stop", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            import importlib
            importlib.reload(app)
        assert err_mock.call_count >= 1
        assert any("Cannot connect to backend" in str(call.args[0]) for call in err_mock.mock_calls)
