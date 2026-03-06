import pytest
import builtins
import types
import sys

import frontend.app as app

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit functions used in app.py to MagicMock
    st_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    monkeypatch.setitem(sys.modules, "st", st_mock)
    # Patch st.session_state to a dict-like object
    st_mock.session_state = {}
    # Patch st.sidebar context manager
    st_mock.sidebar.__enter__.return_value = st_mock.sidebar
    st_mock.sidebar.__exit__.return_value = None
    # Patch st.tabs to return list of MagicMocks
    st_mock.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    # Patch st.columns to return two MagicMocks
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    # Patch st.expander context manager
    st_mock.expander.__enter__.return_value = st_mock.expander
    st_mock.expander.__exit__.return_value = None
    # Patch st.spinner context manager
    st_mock.spinner.__enter__.return_value = st_mock.spinner
    st_mock.spinner.__exit__.return_value = None
    # Patch st.chat_message context manager
    st_mock.chat_message.__enter__.return_value = st_mock.chat_message
    st_mock.chat_message.__exit__.return_value = None
    # Patch st.stop to raise SystemExit
    st_mock.stop.side_effect = SystemExit
    # Patch st.rerun to raise SystemExit
    st_mock.rerun.side_effect = SystemExit
    # Patch st.file_uploader to return None by default
    st_mock.file_uploader.return_value = None
    # Patch st.button to return False by default
    st_mock.button.return_value = False
    # Patch st.chat_input to return None by default
    st_mock.chat_input.return_value = None
    # Patch st.text_input to return default value
    st_mock.text_input.side_effect = lambda label, value, help=None: value
    # Patch st.json to do nothing
    st_mock.json.return_value = None
    # Patch st.caption, st.write, st.markdown, st.success, st.warning, st.error to do nothing
    for fn in ["caption", "write", "markdown", "success", "warning", "error"]:
        getattr(st_mock, fn).return_value = None
    # Patch st.header, st.subheader, st.divider, st.title to do nothing
    for fn in ["header", "subheader", "divider", "title"]:
        getattr(st_mock, fn).return_value = None
    yield

@pytest.fixture
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    with patch("frontend.app.requests.get") as mock_get, patch("frontend.app.requests.post") as mock_post:
        yield mock_get, mock_post

def test_check_backend_happy_path(patch_requests):
    mock_get, _ = patch_requests
    # Simulate backend returns 200
    mock_get.return_value.status_code = 200
    assert app.check_backend() is True
    mock_get.assert_called_once()

def test_check_backend_unreachable(patch_requests):
    mock_get, _ = patch_requests
    # Simulate backend raises exception
    mock_get.side_effect = Exception("Connection error")
    assert app.check_backend() is False
    mock_get.assert_called_once()

def test_check_backend_non_200(patch_requests):
    mock_get, _ = patch_requests
    # Simulate backend returns 500
    mock_get.return_value.status_code = 500
    assert app.check_backend() is False
    mock_get.assert_called_once()

def test_upload_and_index_documents_success(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    # Simulate uploaded files
    fake_file = MagicMock()
    fake_file.name = "test.pdf"
    fake_file.getvalue.return_value = b"pdfdata"
    fake_file.type = "application/pdf"
    # Patch st.file_uploader to return a list with one file
    with patch("frontend.app.st.file_uploader", return_value=[fake_file]):
        # Patch st.button to return True (button pressed)
        with patch("frontend.app.st.button", side_effect=[True]):
            # Patch st.spinner context manager
            with patch("frontend.app.st.spinner"):
                # Simulate backend returns 200 with expected JSON
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "message": "Indexed successfully",
                    "extractions": [
                        {"filename": "test.pdf", "structured_data_extracted": True, "text_chunks": 5}
                    ],
                    "errors": []
                }
                # Should not raise
                import importlib
                importlib.reload(app)

def test_upload_and_index_documents_no_files(monkeypatch):
    # Patch st.file_uploader to return empty list
    with patch("frontend.app.st.file_uploader", return_value=[]):
        # Patch st.button to return True (button pressed)
        with patch("frontend.app.st.button", side_effect=[True]):
            import importlib
            importlib.reload(app)

def test_upload_and_index_documents_backend_error(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    fake_file = MagicMock()
    fake_file.name = "test.pdf"
    fake_file.getvalue.return_value = b"pdfdata"
    fake_file.type = "application/pdf"
    with patch("frontend.app.st.file_uploader", return_value=[fake_file]):
        with patch("frontend.app.st.button", side_effect=[True]):
            with patch("frontend.app.st.spinner"):
                # Simulate backend returns 400
                mock_post.return_value.status_code = 400
                mock_post.return_value.text = "Bad request"
                import importlib
                importlib.reload(app)

def test_upload_and_index_documents_request_exception(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    fake_file = MagicMock()
    fake_file.name = "test.pdf"
    fake_file.getvalue.return_value = b"pdfdata"
    fake_file.type = "application/pdf"
    with patch("frontend.app.st.file_uploader", return_value=[fake_file]):
        with patch("frontend.app.st.button", side_effect=[True]):
            with patch("frontend.app.st.spinner"):
                # Simulate requests.post raises exception
                mock_post.side_effect = Exception("Network error")
                import importlib
                importlib.reload(app)

def test_chat_qa_happy_path(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    # Patch st.chat_input to simulate user input
    with patch("frontend.app.st.chat_input", return_value="What is the rate?"):
        # Patch st.session_state to simulate chat history
        with patch("frontend.app.st.session_state", {}):
            # Patch st.button to return False (not pressed)
            with patch("frontend.app.st.button", return_value=False):
                # Patch st.chat_message context manager
                with patch("frontend.app.st.chat_message"):
                    # Patch st.spinner context manager
                    with patch("frontend.app.st.spinner"):
                        # Simulate backend returns 200 with answer
                        mock_post.return_value.status_code = 200
                        mock_post.return_value.json.return_value = {
                            "answer": "The rate is $1000.",
                            "confidence_score": 0.98,
                            "sources": [
                                {"metadata": {"source": "test.pdf"}, "text": "Relevant text"}
                            ]
                        }
                        import importlib
                        importlib.reload(app)

def test_chat_qa_backend_error(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    with patch("frontend.app.st.chat_input", return_value="What is the rate?"):
        with patch("frontend.app.st.session_state", {}):
            with patch("frontend.app.st.button", return_value=False):
                with patch("frontend.app.st.chat_message"):
                    with patch("frontend.app.st.spinner"):
                        # Simulate backend returns 400
                        mock_post.return_value.status_code = 400
                        mock_post.return_value.text = "Bad request"
                        import importlib
                        importlib.reload(app)

def test_chat_qa_request_exception(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    with patch("frontend.app.st.chat_input", return_value="What is the rate?"):
        with patch("frontend.app.st.session_state", {}):
            with patch("frontend.app.st.button", return_value=False):
                with patch("frontend.app.st.chat_message"):
                    with patch("frontend.app.st.spinner"):
                        # Simulate requests.post raises exception
                        mock_post.side_effect = Exception("Network error")
                        import importlib
                        importlib.reload(app)

def test_data_extraction_success(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    fake_file = MagicMock()
    fake_file.name = "extract.pdf"
    fake_file.getvalue.return_value = b"pdfdata"
    fake_file.type = "application/pdf"
    # Patch st.file_uploader to return a file
    with patch("frontend.app.st.file_uploader", return_value=fake_file):
        # Patch st.button to return True (button pressed)
        with patch("frontend.app.st.button", side_effect=[False, True]):
            with patch("frontend.app.st.spinner"):
                # Simulate backend returns 200 with data
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "data": {
                        "reference_id": "REF123",
                        "load_id": "LOAD456",
                        "shipper": "Shipper Inc.",
                        "consignee": "Consignee LLC",
                        "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
                        "driver": {"driver_name": "John Doe", "truck_number": "TRK123"},
                        "pickup": {"city": "CityA", "state": "ST"},
                        "shipping_date": "2024-01-01",
                        "drop": {"city": "CityB", "state": "ST"},
                        "delivery_date": "2024-01-02",
                        "rate_info": {"total_rate": 1000, "currency": "USD"},
                        "equipment_type": "Van"
                    }
                }
                import importlib
                importlib.reload(app)

def test_data_extraction_backend_error(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    fake_file = MagicMock()
    fake_file.name = "extract.pdf"
    fake_file.getvalue.return_value = b"pdfdata"
    fake_file.type = "application/pdf"
    with patch("frontend.app.st.file_uploader", return_value=fake_file):
        with patch("frontend.app.st.button", side_effect=[False, True]):
            with patch("frontend.app.st.spinner"):
                # Simulate backend returns 400
                mock_post.return_value.status_code = 400
                mock_post.return_value.text = "Bad request"
                import importlib
                importlib.reload(app)

def test_data_extraction_request_exception(monkeypatch, patch_requests):
    _, mock_post = patch_requests
    fake_file = MagicMock()
    fake_file.name = "extract.pdf"
    fake_file.getvalue.return_value = b"pdfdata"
    fake_file.type = "application/pdf"
    with patch("frontend.app.st.file_uploader", return_value=fake_file):
        with patch("frontend.app.st.button", side_effect=[False, True]):
            with patch("frontend.app.st.spinner"):
                # Simulate requests.post raises exception
                mock_post.side_effect = Exception("Network error")
                import importlib
                importlib.reload(app)

def test_data_extraction_no_file(monkeypatch):
    # Patch st.file_uploader to return None
    with patch("frontend.app.st.file_uploader", return_value=None):
        with patch("frontend.app.st.button", side_effect=[False, True]):
            import importlib
            importlib.reload(app)

def test_backend_unreachable(monkeypatch, patch_requests):
    # Simulate backend is unreachable
    mock_get, _ = patch_requests
    mock_get.side_effect = Exception("Connection error")
    # Patch st.button to return False (Retry not pressed)
    with patch("frontend.app.st.button", return_value=False):
        # Patch st.stop to raise SystemExit
        with pytest.raises(SystemExit):
            import importlib
            importlib.reload(app)

def test_backend_unreachable_retry(monkeypatch, patch_requests):
    # Simulate backend is unreachable
    mock_get, _ = patch_requests
    mock_get.side_effect = Exception("Connection error")
    # Patch st.button to return True (Retry pressed)
    with patch("frontend.app.st.button", return_value=True):
        # Patch st.rerun to raise SystemExit
        with pytest.raises(SystemExit):
            import importlib
            importlib.reload(app)
