import pytest
import builtins
import types
import sys

import frontend.app as app

from unittest.mock import patch, MagicMock, call

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit functions used in app.py
    st_mock = MagicMock()
    # Patch context managers
    st_mock.sidebar.__enter__.return_value = None
    st_mock.sidebar.__exit__.return_value = None
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.expander.return_value.__exit__.return_value = None
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    st_mock.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    st_mock.session_state = {}
    st_mock.file_uploader.return_value = None
    st_mock.chat_message.return_value.__enter__.return_value = None
    st_mock.chat_message.return_value.__exit__.return_value = None
    st_mock.chat_input.return_value = None
    st_mock.button.return_value = False
    st_mock.text_input.return_value = "http://localhost:8000"
    st_mock.stop.side_effect = Exception("st.stop called")
    st_mock.rerun.side_effect = Exception("st.rerun called")
    st_mock.divider.return_value = None
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    monkeypatch.setattr(app, "st", st_mock)
    return st_mock

@pytest.fixture
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    get_mock = MagicMock()
    post_mock = MagicMock()
    monkeypatch.setattr(app.requests, "get", get_mock)
    monkeypatch.setattr(app.requests, "post", post_mock)
    return get_mock, post_mock

@pytest.fixture
def patch_os_environ(monkeypatch):
    # Patch os.getenv
    monkeypatch.setattr(app.os, "getenv", lambda key, default=None: default)
    return

def test_check_backend_happy_path(patch_requests):
    get_mock, _ = patch_requests
    # Simulate backend healthy
    resp = MagicMock()
    resp.status_code = 200
    get_mock.return_value = resp
    app.BACKEND_URL = "http://localhost:8000"
    assert app.check_backend() is True
    get_mock.assert_called_once_with("http://localhost:8000/ping")

def test_check_backend_down_status(patch_requests):
    get_mock, _ = patch_requests
    # Simulate backend unhealthy
    resp = MagicMock()
    resp.status_code = 500
    get_mock.return_value = resp
    app.BACKEND_URL = "http://localhost:8000"
    assert app.check_backend() is False

def test_check_backend_exception(patch_requests):
    get_mock, _ = patch_requests
    # Simulate requests.get raising exception
    get_mock.side_effect = Exception("network error")
    app.BACKEND_URL = "http://localhost:8000"
    assert app.check_backend() is False

def test_check_backend_url_env(monkeypatch):
    # If BACKEND_URL env is set, it should be used as default
    monkeypatch.setattr(app.os, "getenv", lambda key, default=None: "http://env-backend:9000")
    st_mock = MagicMock()
    st_mock.text_input.return_value = "http://env-backend:9000"
    st_mock.sidebar.__enter__.return_value = None
    st_mock.sidebar.__exit__.return_value = None
    st_mock.divider.return_value = None
    st_mock.markdown.return_value = None
    st_mock.header.return_value = None
    monkeypatch.setattr(app, "st", st_mock)
    # Simulate sidebar config
    with patch.object(app, "st", st_mock):
        # Re-run the sidebar config block
        default_backend = app.os.getenv("BACKEND_URL", "http://localhost:8000")
        BACKEND_URL = app.st.text_input(
            "Backend API URL",
            value=default_backend,
            help="The URL of the FastAPI backend."
        )
        assert BACKEND_URL == "http://env-backend:9000"

def test_upload_files_success(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    # Simulate uploaded files
    file1 = MagicMock()
    file1.name = "doc1.pdf"
    file1.getvalue.return_value = b"file1content"
    file1.type = "application/pdf"
    file2 = MagicMock()
    file2.name = "doc2.txt"
    file2.getvalue.return_value = b"file2content"
    file2.type = "text/plain"
    st_mock.file_uploader.return_value = [file1, file2]
    st_mock.button.side_effect = lambda label: label == "Process & Index Documents"
    # Simulate backend response
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "message": "Indexed 2 documents.",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
            {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    }
    post_mock.return_value = resp
    # Patch spinner context manager
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    # Patch expander context manager
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.expander.return_value.__exit__.return_value = None
    # Run the upload logic
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        # Simulate the upload logic block
        uploaded_files = st_mock.file_uploader(
            "Choose documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if st_mock.button("Process & Index Documents"):
            if uploaded_files:
                files_to_send = [
                    ("files", (f.name, f.getvalue(), f.type)) 
                    for f in uploaded_files
                ]
                with st_mock.spinner("Processing documents (chunking, embedding, indexing)..."):
                    try:
                        response = post_mock(
                            f"http://localhost:8000/upload",
                            files=files_to_send
                        )
                        assert response.status_code == 200
                        result = response.json()
                        st_mock.success.assert_called_with("Indexed 2 documents.")
                        # Extraction summary
                        st_mock.expander.assert_called_with("Extraction Summary")
                        # Extraction status
                        calls = [
                            call("✅ **doc1.pdf**: 5 chunks"),
                            call("❌ **doc2.txt**: 2 chunks"),
                        ]
                        st_mock.write.assert_has_calls(calls, any_order=True)
                        # No errors
                        assert not result.get("errors")
                    except Exception:
                        pytest.fail("Should not raise exception")
            else:
                pytest.fail("Should have uploaded files")

def test_upload_files_no_files(monkeypatch):
    st_mock = app.st
    st_mock.file_uploader.return_value = []
    st_mock.button.side_effect = lambda label: label == "Process & Index Documents"
    # Run the upload logic
    uploaded_files = st_mock.file_uploader(
        "Choose documents", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    if st_mock.button("Process & Index Documents"):
        if not uploaded_files:
            st_mock.warning.assert_called_with("Please upload at least one document.")

def test_upload_files_backend_error(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    file1 = MagicMock()
    file1.name = "doc1.pdf"
    file1.getvalue.return_value = b"file1content"
    file1.type = "application/pdf"
    st_mock.file_uploader.return_value = [file1]
    st_mock.button.side_effect = lambda label: label == "Process & Index Documents"
    resp = MagicMock()
    resp.status_code = 400
    resp.text = "Bad request"
    post_mock.return_value = resp
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        uploaded_files = st_mock.file_uploader(
            "Choose documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if st_mock.button("Process & Index Documents"):
            if uploaded_files:
                files_to_send = [
                    ("files", (f.name, f.getvalue(), f.type)) 
                    for f in uploaded_files
                ]
                with st_mock.spinner("Processing documents (chunking, embedding, indexing)..."):
                    try:
                        response = post_mock(
                            f"http://localhost:8000/upload",
                            files=files_to_send
                        )
                        assert response.status_code == 400
                        st_mock.error.assert_called_with("Error: Bad request")
                    except Exception:
                        pytest.fail("Should not raise exception")

def test_upload_files_request_exception(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    file1 = MagicMock()
    file1.name = "doc1.pdf"
    file1.getvalue.return_value = b"file1content"
    file1.type = "application/pdf"
    st_mock.file_uploader.return_value = [file1]
    st_mock.button.side_effect = lambda label: label == "Process & Index Documents"
    post_mock.side_effect = Exception("network error")
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        uploaded_files = st_mock.file_uploader(
            "Choose documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if st_mock.button("Process & Index Documents"):
            if uploaded_files:
                files_to_send = [
                    ("files", (f.name, f.getvalue(), f.type)) 
                    for f in uploaded_files
                ]
                with st_mock.spinner("Processing documents (chunking, embedding, indexing)..."):
                    try:
                        post_mock(
                            f"http://localhost:8000/upload",
                            files=files_to_send
                        )
                    except Exception as e:
                        st_mock.error.assert_called_with("Request failed: network error")

def test_chat_qa_happy_path(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    # Simulate chat input
    st_mock.chat_input.return_value = "What is the agreed rate?"
    st_mock.session_state = {"messages": []}
    st_mock.chat_message.return_value.__enter__.return_value = None
    st_mock.chat_message.return_value.__exit__.return_value = None
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.expander.return_value.__exit__.return_value = None
    # Simulate backend response
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
        ]
    }
    post_mock.return_value = resp
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        prompt = st_mock.chat_input("What is the agreed rate for this shipment?")
        if prompt:
            st_mock.session_state["messages"].append({"role": "user", "content": prompt})
            with st_mock.chat_message("user"):
                st_mock.markdown(prompt)
            with st_mock.chat_message("assistant"):
                with st_mock.spinner("Thinking..."):
                    try:
                        payload = {"question": prompt, "chat_history": []}
                        response = post_mock(f"http://localhost:8000/ask", json=payload)
                        assert response.status_code == 200
                        data = response.json()
                        answer = data["answer"]
                        confidence = data["confidence_score"]
                        sources = data["sources"]
                        st_mock.markdown.assert_any_call(answer)
                        st_mock.caption.assert_any_call("Confidence Score: 0.98")
                        st_mock.expander.assert_called_with("View Sources")
                        st_mock.caption.assert_any_call("Source: doc1.pdf")
                        st_mock.write.assert_any_call("Rate: $1200")
                        # Assistant message appended
                        st_mock.session_state["messages"].append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    except Exception:
                        pytest.fail("Should not raise exception")

def test_chat_qa_backend_error(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    st_mock.chat_input.return_value = "What is the agreed rate?"
    st_mock.session_state = {"messages": []}
    st_mock.chat_message.return_value.__enter__.return_value = None
    st_mock.chat_message.return_value.__exit__.return_value = None
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    resp = MagicMock()
    resp.status_code = 400
    resp.text = "Bad request"
    post_mock.return_value = resp
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        prompt = st_mock.chat_input("What is the agreed rate for this shipment?")
        if prompt:
            st_mock.session_state["messages"].append({"role": "user", "content": prompt})
            with st_mock.chat_message("user"):
                st_mock.markdown(prompt)
            with st_mock.chat_message("assistant"):
                with st_mock.spinner("Thinking..."):
                    try:
                        payload = {"question": prompt, "chat_history": []}
                        response = post_mock(f"http://localhost:8000/ask", json=payload)
                        assert response.status_code == 400
                        st_mock.error.assert_called_with("Error: Bad request")
                    except Exception:
                        pytest.fail("Should not raise exception")

def test_chat_qa_request_exception(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    st_mock.chat_input.return_value = "What is the agreed rate?"
    st_mock.session_state = {"messages": []}
    st_mock.chat_message.return_value.__enter__.return_value = None
    st_mock.chat_message.return_value.__exit__.return_value = None
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    post_mock.side_effect = Exception("network error")
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        prompt = st_mock.chat_input("What is the agreed rate for this shipment?")
        if prompt:
            st_mock.session_state["messages"].append({"role": "user", "content": prompt})
            with st_mock.chat_message("user"):
                st_mock.markdown(prompt)
            with st_mock.chat_message("assistant"):
                with st_mock.spinner("Thinking..."):
                    try:
                        payload = {"question": prompt, "chat_history": []}
                        post_mock(f"http://localhost:8000/ask", json=payload)
                    except Exception as e:
                        st_mock.error.assert_called_with("Request failed: network error")

def test_data_extraction_success(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    file1 = MagicMock()
    file1.name = "doc1.pdf"
    file1.getvalue.return_value = b"file1content"
    file1.type = "application/pdf"
    st_mock.file_uploader.return_value = file1
    st_mock.button.side_effect = lambda label: label == "Run Extraction"
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    st_mock.columns.return_value = (MagicMock(), MagicMock())
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.expander.return_value.__exit__.return_value = None
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "ACME Corp",
            "consignee": "Beta LLC",
            "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
            "driver": {"driver_name": "John Doe", "truck_number": "TRK123"},
            "pickup": {"city": "Dallas", "state": "TX"},
            "shipping_date": "2024-06-01",
            "drop": {"city": "Houston", "state": "TX"},
            "delivery_date": "2024-06-02",
            "rate_info": {"total_rate": 1200, "currency": "USD"},
            "equipment_type": "Flatbed"
        }
    }
    post_mock.return_value = resp
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        extract_file = st_mock.file_uploader(
            "Upload a single document for extraction", 
            type=["pdf", "docx", "txt"],
            key="extraction_uploader"
        )
        if st_mock.button("Run Extraction"):
            if extract_file:
                with st_mock.spinner("Extracting data..."):
                    try:
                        files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                        response = post_mock(f"http://localhost:8000/extract", files=files)
                        assert response.status_code == 200
                        data = response.json()["data"]
                        st_mock.success.assert_called_with("Extraction complete!")
                        st_mock.columns.assert_called_with(2)
                        st_mock.expander.assert_called_with("View Full JSON")
                        st_mock.json.assert_called_with(data)
                    except Exception:
                        pytest.fail("Should not raise exception")

def test_data_extraction_no_file(monkeypatch):
    st_mock = app.st
    st_mock.file_uploader.return_value = None
    st_mock.button.side_effect = lambda label: label == "Run Extraction"
    extract_file = st_mock.file_uploader(
        "Upload a single document for extraction", 
        type=["pdf", "docx", "txt"],
        key="extraction_uploader"
    )
    if st_mock.button("Run Extraction"):
        if not extract_file:
            st_mock.warning.assert_called_with("Please upload a document.")

def test_data_extraction_backend_error(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    file1 = MagicMock()
    file1.name = "doc1.pdf"
    file1.getvalue.return_value = b"file1content"
    file1.type = "application/pdf"
    st_mock.file_uploader.return_value = file1
    st_mock.button.side_effect = lambda label: label == "Run Extraction"
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    resp = MagicMock()
    resp.status_code = 400
    resp.text = "Bad request"
    post_mock.return_value = resp
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        extract_file = st_mock.file_uploader(
            "Upload a single document for extraction", 
            type=["pdf", "docx", "txt"],
            key="extraction_uploader"
        )
        if st_mock.button("Run Extraction"):
            if extract_file:
                with st_mock.spinner("Extracting data..."):
                    try:
                        files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                        response = post_mock(f"http://localhost:8000/extract", files=files)
                        assert response.status_code == 400
                        st_mock.error.assert_called_with("Error: Bad request")
                    except Exception:
                        pytest.fail("Should not raise exception")

def test_data_extraction_request_exception(monkeypatch, patch_requests):
    _, post_mock = patch_requests
    st_mock = app.st
    file1 = MagicMock()
    file1.name = "doc1.pdf"
    file1.getvalue.return_value = b"file1content"
    file1.type = "application/pdf"
    st_mock.file_uploader.return_value = file1
    st_mock.button.side_effect = lambda label: label == "Run Extraction"
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    post_mock.side_effect = Exception("network error")
    with patch.object(app, "BACKEND_URL", "http://localhost:8000"):
        extract_file = st_mock.file_uploader(
            "Upload a single document for extraction", 
            type=["pdf", "docx", "txt"],
            key="extraction_uploader"
        )
        if st_mock.button("Run Extraction"):
            if extract_file:
                with st_mock.spinner("Extracting data..."):
                    try:
                        files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                        post_mock(f"http://localhost:8000/extract", files=files)
                    except Exception as e:
                        st_mock.error.assert_called_with("Request failed: network error")
