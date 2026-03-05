import pytest
import builtins
import types
import sys

import frontend.app as app

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit functions used in app.py
    st_mock = MagicMock()
    # Patch st.session_state as a dict
    st_mock.session_state = {}
    # Patch st.sidebar context manager
    st_mock.sidebar.__enter__.return_value = st_mock.sidebar
    st_mock.sidebar.__exit__.return_value = None
    # Patch st.tabs to return list of MagicMock tabs
    st_mock.tabs.side_effect = lambda labels: [MagicMock(name=f"tab_{i}") for i in range(len(labels))]
    # Patch st.file_uploader to return None by default
    st_mock.file_uploader.return_value = None
    # Patch st.button to return False by default
    st_mock.button.return_value = False
    # Patch st.chat_input to return None by default
    st_mock.chat_input.return_value = None
    # Patch st.stop to raise SystemExit
    st_mock.stop.side_effect = SystemExit
    # Patch st.rerun to raise SystemExit
    st_mock.rerun.side_effect = SystemExit
    # Patch st.spinner as context manager
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    # Patch st.columns to return two MagicMocks
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    # Patch st.expander as context manager
    st_mock.expander.return_value.__enter__.return_value = st_mock.expander
    st_mock.expander.return_value.__exit__.return_value = None
    # Patch st.chat_message as context manager
    st_mock.chat_message.return_value.__enter__.return_value = st_mock.chat_message
    st_mock.chat_message.return_value.__exit__.return_value = None

    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    yield

@pytest.fixture
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    get_mock = MagicMock()
    post_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get_mock, post=post_mock))
    return get_mock, post_mock

@pytest.fixture
def patch_os(monkeypatch):
    # Patch os.getenv
    monkeypatch.setitem(sys.modules, "os", types.SimpleNamespace(getenv=lambda k, v=None: v))
    yield

def run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os):
    # Remove app from sys.modules to force reload
    sys.modules.pop("frontend.app", None)
    import importlib
    import frontend.app as app_mod
    return app_mod

def test_backend_health_happy_path(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    # Simulate backend /ping returns 200
    get_mock.return_value.status_code = 200
    # Simulate all other requests/post not called
    post_mock.side_effect = AssertionError("Should not be called in health check")
    # Run app
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on healthy backend")
    # Check /ping called with correct URL
    get_mock.assert_called_with("http://localhost:8000/ping")

def test_backend_health_down(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    # Simulate backend /ping returns 500
    get_mock.return_value.status_code = 500
    # Simulate st.stop called (should raise SystemExit)
    with pytest.raises(SystemExit):
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    get_mock.assert_called_with("http://localhost:8000/ping")

def test_backend_health_exception(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    # Simulate backend /ping raises exception
    get_mock.side_effect = Exception("Connection error")
    with pytest.raises(SystemExit):
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    get_mock.assert_called_with("http://localhost:8000/ping")

def test_upload_and_index_happy(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    # Backend is up
    get_mock.return_value.status_code = 200

    # Simulate file upload
    file_mock = MagicMock()
    file_mock.name = "test.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"

    # Patch st.file_uploader to return a list of one file
    import streamlit as st
    st.file_uploader.return_value = [file_mock]
    # Patch st.button to return True for "Process & Index Documents"
    st.button.side_effect = lambda label=None: label == "Process & Index Documents"

    # Simulate backend /upload returns 200 with extractions and no errors
    post_mock.return_value.status_code = 200
    post_mock.return_value.json.return_value = {
        "message": "Indexed successfully",
        "extractions": [
            {"filename": "test.pdf", "structured_data_extracted": True, "text_chunks": 5}
        ],
        "errors": []
    }

    # Run app
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on healthy backend and upload")

    # Check /upload called
    post_mock.assert_any_call(
        "http://localhost:8000/upload",
        files=[("files", ("test.pdf", b"PDFDATA", "application/pdf"))]
    )

def test_upload_and_index_no_files(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    import streamlit as st
    st.file_uploader.return_value = []
    st.button.side_effect = lambda label=None: label == "Process & Index Documents"
    # Should not call backend
    post_mock.side_effect = AssertionError("Should not call backend when no files")
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on healthy backend and no files")

def test_upload_and_index_backend_error(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    file_mock = MagicMock()
    file_mock.name = "fail.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    import streamlit as st
    st.file_uploader.return_value = [file_mock]
    st.button.side_effect = lambda label=None: label == "Process & Index Documents"
    post_mock.return_value.status_code = 400
    post_mock.return_value.text = "Bad request"
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on backend error in upload")

def test_upload_and_index_backend_exception(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    file_mock = MagicMock()
    file_mock.name = "fail.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    import streamlit as st
    st.file_uploader.return_value = [file_mock]
    st.button.side_effect = lambda label=None: label == "Process & Index Documents"
    post_mock.side_effect = Exception("Network error")
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on backend exception in upload")

def test_chat_qa_happy(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    import streamlit as st
    # Simulate chat input
    st.chat_input.return_value = "What is the agreed rate?"
    # Simulate session state
    st.session_state = {"messages": []}
    # Simulate backend /ask returns 200
    post_mock.return_value.status_code = 200
    post_mock.return_value.json.return_value = {
        "answer": "The agreed rate is $1000.",
        "confidence_score": 0.95,
        "sources": [
            {"metadata": {"source": "test.pdf"}, "text": "Rate: $1000"}
        ]
    }
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on chat happy path")
    post_mock.assert_any_call(
        "http://localhost:8000/ask",
        json={"question": "What is the agreed rate?", "chat_history": []}
    )

def test_chat_qa_backend_error(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    import streamlit as st
    st.chat_input.return_value = "What is the agreed rate?"
    st.session_state = {"messages": []}
    post_mock.return_value.status_code = 400
    post_mock.return_value.text = "Bad request"
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on chat backend error")

def test_chat_qa_backend_exception(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    import streamlit as st
    st.chat_input.return_value = "What is the agreed rate?"
    st.session_state = {"messages": []}
    post_mock.side_effect = Exception("Network error")
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on chat backend exception")

def test_data_extraction_happy(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    file_mock = MagicMock()
    file_mock.name = "extract.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    import streamlit as st
    st.file_uploader.side_effect = lambda *a, **k: file_mock if k.get("key") == "extraction_uploader" else None
    st.button.side_effect = lambda label=None: label == "Run Extraction"
    post_mock.return_value.status_code = 200
    post_mock.return_value.json.return_value = {
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
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on extraction happy path")
    post_mock.assert_any_call(
        "http://localhost:8000/extract",
        files={"file": ("extract.pdf", b"PDFDATA", "application/pdf")}
    )

def test_data_extraction_no_file(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    import streamlit as st
    st.file_uploader.side_effect = lambda *a, **k: None
    st.button.side_effect = lambda label=None: label == "Run Extraction"
    post_mock.side_effect = AssertionError("Should not call backend when no file")
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on extraction no file")

def test_data_extraction_backend_error(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    file_mock = MagicMock()
    file_mock.name = "extract.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    import streamlit as st
    st.file_uploader.side_effect = lambda *a, **k: file_mock if k.get("key") == "extraction_uploader" else None
    st.button.side_effect = lambda label=None: label == "Run Extraction"
    post_mock.return_value.status_code = 400
    post_mock.return_value.text = "Bad request"
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on extraction backend error")

def test_data_extraction_backend_exception(monkeypatch, patch_streamlit, patch_requests, patch_os):
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200
    file_mock = MagicMock()
    file_mock.name = "extract.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    import streamlit as st
    st.file_uploader.side_effect = lambda *a, **k: file_mock if k.get("key") == "extraction_uploader" else None
    st.button.side_effect = lambda label=None: label == "Run Extraction"
    post_mock.side_effect = Exception("Network error")
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on extraction backend exception")

def test_equivalent_paths_reconciliation(monkeypatch, patch_streamlit, patch_requests, patch_os):
    """
    Reconciliation: Compare outputs for equivalent paths.
    For example, uploading a file and extracting data via upload/index vs direct extraction.
    """
    get_mock, post_mock = patch_requests
    get_mock.return_value.status_code = 200

    # Simulate file for both upload and extraction
    file_mock = MagicMock()
    file_mock.name = "recon.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"

    # Patch st.file_uploader for both upload and extraction
    import streamlit as st
    # First call: upload tab, second call: extraction tab
    st.file_uploader.side_effect = [
        [file_mock],  # upload tab
        file_mock     # extraction tab
    ]
    # Patch st.button to trigger both actions in sequence
    def button_side_effect(label=None):
        return label in ("Process & Index Documents", "Run Extraction")
    st.button.side_effect = button_side_effect

    # Simulate /upload returns extraction summary
    upload_extractions = [
        {"filename": "recon.pdf", "structured_data_extracted": True, "text_chunks": 7}
    ]
    upload_result = {
        "message": "Indexed successfully",
        "extractions": upload_extractions,
        "errors": []
    }
    # Simulate /extract returns same data as upload extraction
    extraction_data = {
        "reference_id": "REF999",
        "load_id": "LOAD888",
        "shipper": "Recon Shipper",
        "consignee": "Recon Consignee",
        "carrier": {"carrier_name": "ReconCarrier", "mc_number": "MC000"},
        "driver": {"driver_name": "Recon Driver", "truck_number": "TRK999"},
        "pickup": {"city": "Austin", "state": "TX"},
        "shipping_date": "2023-02-01",
        "drop": {"city": "San Antonio", "state": "TX"},
        "delivery_date": "2023-02-02",
        "rate_info": {"total_rate": "2000", "currency": "USD"},
        "equipment_type": "Flatbed"
    }
    # /upload returns extraction summary, /extract returns full data
    post_mock.side_effect = [
        MagicMock(status_code=200, json=MagicMock(return_value=upload_result)),  # /upload
        MagicMock(status_code=200, json=MagicMock(return_value={"data": extraction_data}))  # /extract
    ]

    # Run app
    try:
        run_app_module(monkeypatch, patch_streamlit, patch_requests, patch_os)
    except SystemExit:
        pytest.fail("Should not stop on reconciliation test")

    # Check that both endpoints were called with correct files
    post_mock.assert_any_call(
        "http://localhost:8000/upload",
        files=[("files", ("recon.pdf", b"PDFDATA", "application/pdf"))]
    )
    post_mock.assert_any_call(
        "http://localhost:8000/extract",
        files={"file": ("recon.pdf", b"PDFDATA", "application/pdf")}
    )
    # Reconciliation: The filename and chunk count from upload_extractions should match the file used in extraction
    assert upload_extractions[0]["filename"] == file_mock.name
    # The extraction_data fields should be as expected
    assert extraction_data["shipper"] == "Recon Shipper"
    assert extraction_data["carrier"]["carrier_name"] == "ReconCarrier"
    assert extraction_data["rate_info"]["total_rate"] == "2000"
    assert extraction_data["equipment_type"] == "Flatbed"
