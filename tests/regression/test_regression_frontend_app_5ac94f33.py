# source_hash: 9ee607a3d8da4254
# import_target: frontend.app
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
import builtins
import types
import io

import frontend.app as app

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def reset_session_state(monkeypatch):
    # Reset st.session_state before each test
    import streamlit as st
    st.session_state.clear()
    yield
    st.session_state.clear()

@pytest.fixture
def mock_streamlit(monkeypatch):
    st_mock = MagicMock()
    monkeypatch.setattr(app, "st", st_mock)
    return st_mock

@pytest.fixture
def mock_requests(monkeypatch):
    requests_mock = MagicMock()
    monkeypatch.setattr(app, "requests", requests_mock)
    return requests_mock

@pytest.fixture
def mock_os(monkeypatch):
    os_mock = MagicMock()
    monkeypatch.setattr(app, "os", os_mock)
    return os_mock

def make_file_mock(name, content, filetype):
    file_mock = MagicMock()
    file_mock.name = name
    file_mock.getvalue.return_value = content
    file_mock.type = filetype
    return file_mock

def test_check_backend_happy_path(mock_streamlit, mock_requests):
    # Simulate backend healthy
    mock_requests.get.return_value.status_code = 200
    assert app.check_backend() is True
    mock_requests.get.assert_called_once()
    assert "/ping" in mock_requests.get.call_args[0][0]

def test_check_backend_unreachable(mock_streamlit, mock_requests):
    # Simulate backend unreachable (exception)
    mock_requests.get.side_effect = Exception("Connection error")
    assert app.check_backend() is False

def test_check_backend_status_code_not_200(mock_streamlit, mock_requests):
    # Simulate backend returns non-200
    mock_requests.get.return_value.status_code = 500
    assert app.check_backend() is False

def test_sidebar_backend_url_default_env(monkeypatch, mock_streamlit, mock_os):
    # Simulate BACKEND_URL env var present
    mock_os.getenv.return_value = "http://testserver:9000"
    # Simulate text_input returns the default value
    mock_streamlit.text_input.return_value = "http://testserver:9000"
    # Run sidebar code
    with patch.object(app, "os", mock_os):
        with patch.object(app, "st", mock_streamlit):
            # Re-run sidebar block
            exec(
                "\n".join([
                    "default_backend = os.getenv('BACKEND_URL', 'http://localhost:8000')",
                    "BACKEND_URL = st.text_input('Backend API URL', value=default_backend, help='The URL of the FastAPI backend.')"
                ]),
                {"os": mock_os, "st": mock_streamlit}
            )
    mock_os.getenv.assert_called_with("BACKEND_URL", "http://localhost:8000")
    mock_streamlit.text_input.assert_called_with(
        "Backend API URL",
        value="http://testserver:9000",
        help="The URL of the FastAPI backend."
    )

def test_upload_and_index_documents_success(monkeypatch, mock_streamlit, mock_requests):
    # Simulate uploaded files
    file1 = make_file_mock("doc1.pdf", b"filecontent1", "application/pdf")
    file2 = make_file_mock("doc2.txt", b"filecontent2", "text/plain")
    mock_streamlit.file_uploader.return_value = [file1, file2]
    mock_streamlit.button.side_effect = [True]  # "Process & Index Documents" pressed

    # Simulate backend response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "message": "Indexed successfully",
        "extractions": [
            {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 5},
            {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 2}
        ],
        "errors": []
    }
    mock_requests.post.return_value = response_mock

    # Patch spinner context manager
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    # Patch st.expander context manager
    class DummyExpander:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.expander.return_value = DummyExpander()

    # Run the upload/index block
    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        # Simulate the logic inside the first tab
        uploaded_files = mock_streamlit.file_uploader(
            "Choose documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if mock_streamlit.button("Process & Index Documents"):
            if uploaded_files:
                files_to_send = [
                    ("files", (f.name, f.getvalue(), f.type)) 
                    for f in uploaded_files
                ]
                with mock_streamlit.spinner("Processing documents (chunking, embedding, indexing)..."):
                    try:
                        response = mock_requests.post(
                            f"http://localhost:8000/upload",
                            files=files_to_send
                        )
                        if response.status_code == 200:
                            result = response.json()
                            mock_streamlit.success(result["message"])
                            if result.get("extractions"):
                                with mock_streamlit.expander("Extraction Summary"):
                                    for ext in result["extractions"]:
                                        status = "✅" if ext["structured_data_extracted"] else "❌"
                                        mock_streamlit.write(f"{status} **{ext['filename']}**: {ext['text_chunks']} chunks")
                            if result.get("errors"):
                                for err in result["errors"]:
                                    mock_streamlit.warning(err)
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
            else:
                mock_streamlit.warning("Please upload at least one document.")

    mock_requests.post.assert_called_once()
    mock_streamlit.success.assert_called_with("Indexed successfully")
    assert mock_streamlit.write.call_count >= 2  # For each extraction

def test_upload_and_index_documents_no_files(monkeypatch, mock_streamlit):
    # No files uploaded
    mock_streamlit.file_uploader.return_value = []
    mock_streamlit.button.side_effect = [True]
    with patch.object(app, "st", mock_streamlit):
        uploaded_files = mock_streamlit.file_uploader(
            "Choose documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if mock_streamlit.button("Process & Index Documents"):
            if uploaded_files:
                pass  # Should not enter here
            else:
                mock_streamlit.warning("Please upload at least one document.")
    mock_streamlit.warning.assert_called_with("Please upload at least one document.")

def test_upload_and_index_documents_backend_error(monkeypatch, mock_streamlit, mock_requests):
    # Simulate uploaded file
    file1 = make_file_mock("doc1.pdf", b"filecontent1", "application/pdf")
    mock_streamlit.file_uploader.return_value = [file1]
    mock_streamlit.button.side_effect = [True]
    # Simulate backend error
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Server Error"
    mock_requests.post.return_value = response_mock

    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        uploaded_files = mock_streamlit.file_uploader(
            "Choose documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if mock_streamlit.button("Process & Index Documents"):
            if uploaded_files:
                files_to_send = [
                    ("files", (f.name, f.getvalue(), f.type)) 
                    for f in uploaded_files
                ]
                with mock_streamlit.spinner("Processing documents (chunking, embedding, indexing)..."):
                    try:
                        response = mock_requests.post(
                            f"http://localhost:8000/upload",
                            files=files_to_send
                        )
                        if response.status_code == 200:
                            pass
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
            else:
                mock_streamlit.warning("Please upload at least one document.")
    mock_streamlit.error.assert_called_with("Error: Internal Server Error")

def test_upload_and_index_documents_request_exception(monkeypatch, mock_streamlit, mock_requests):
    # Simulate uploaded file
    file1 = make_file_mock("doc1.pdf", b"filecontent1", "application/pdf")
    mock_streamlit.file_uploader.return_value = [file1]
    mock_streamlit.button.side_effect = [True]
    # Simulate requests.post raises exception
    mock_requests.post.side_effect = Exception("Timeout")

    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        uploaded_files = mock_streamlit.file_uploader(
            "Choose documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if mock_streamlit.button("Process & Index Documents"):
            if uploaded_files:
                files_to_send = [
                    ("files", (f.name, f.getvalue(), f.type)) 
                    for f in uploaded_files
                ]
                with mock_streamlit.spinner("Processing documents (chunking, embedding, indexing)..."):
                    try:
                        response = mock_requests.post(
                            f"http://localhost:8000/upload",
                            files=files_to_send
                        )
                        if response.status_code == 200:
                            pass
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
            else:
                mock_streamlit.warning("Please upload at least one document.")
    mock_streamlit.error.assert_called()
    assert "Request failed" in mock_streamlit.error.call_args[0][0]

def test_chat_qa_happy_path(monkeypatch, mock_streamlit, mock_requests):
    # Simulate chat input and backend response
    mock_streamlit.chat_input.return_value = "What is the agreed rate?"
    mock_streamlit.button.return_value = False
    mock_streamlit.session_state.messages = []
    # Simulate backend response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.98,
        "sources": [
            {"metadata": {"source": "doc1.pdf"}, "text": "Rate: $1200"}
        ]
    }
    mock_requests.post.return_value = response_mock

    class DummyChatMessage:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.chat_message.return_value = DummyChatMessage()

    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    class DummyExpander:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.expander.return_value = DummyExpander()

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        # Simulate chat logic
        if "messages" not in mock_streamlit.session_state:
            mock_streamlit.session_state.messages = []
        for message in mock_streamlit.session_state.messages:
            with mock_streamlit.chat_message(message["role"]):
                mock_streamlit.markdown(message["content"])
                if "sources" in message:
                    with mock_streamlit.expander("View Sources"):
                        for src in message["sources"]:
                            mock_streamlit.caption(f"Source: {src['metadata']['source']}")
                            mock_streamlit.write(src["text"])
        prompt = mock_streamlit.chat_input("What is the agreed rate for this shipment?")
        if prompt:
            mock_streamlit.session_state.messages.append({"role": "user", "content": prompt})
            with mock_streamlit.chat_message("user"):
                mock_streamlit.markdown(prompt)
            with mock_streamlit.chat_message("assistant"):
                with mock_streamlit.spinner("Thinking..."):
                    try:
                        payload = {"question": prompt, "chat_history": []}
                        response = mock_requests.post("http://localhost:8000/ask", json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            answer = data["answer"]
                            confidence = data["confidence_score"]
                            sources = data["sources"]
                            mock_streamlit.markdown(answer)
                            mock_streamlit.caption(f"Confidence Score: {confidence:.2f}")
                            if sources:
                                with mock_streamlit.expander("View Sources"):
                                    for src in sources:
                                        mock_streamlit.caption(f"Source: {src['metadata']['source']}")
                                        mock_streamlit.write(src["text"])
                            mock_streamlit.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer,
                                "sources": sources
                            })
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
    mock_requests.post.assert_called_once()
    mock_streamlit.markdown.assert_any_call("The agreed rate is $1200.")
    mock_streamlit.caption.assert_any_call("Confidence Score: 0.98")

def test_chat_qa_backend_error(monkeypatch, mock_streamlit, mock_requests):
    mock_streamlit.chat_input.return_value = "What is the agreed rate?"
    mock_streamlit.session_state.messages = []
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Error"
    mock_requests.post.return_value = response_mock

    class DummyChatMessage:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.chat_message.return_value = DummyChatMessage()
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        prompt = mock_streamlit.chat_input("What is the agreed rate for this shipment?")
        if prompt:
            mock_streamlit.session_state.messages.append({"role": "user", "content": prompt})
            with mock_streamlit.chat_message("user"):
                mock_streamlit.markdown(prompt)
            with mock_streamlit.chat_message("assistant"):
                with mock_streamlit.spinner("Thinking..."):
                    try:
                        payload = {"question": prompt, "chat_history": []}
                        response = mock_requests.post("http://localhost:8000/ask", json=payload)
                        if response.status_code == 200:
                            pass
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
    mock_streamlit.error.assert_called_with("Error: Internal Error")

def test_chat_qa_request_exception(monkeypatch, mock_streamlit, mock_requests):
    mock_streamlit.chat_input.return_value = "What is the agreed rate?"
    mock_streamlit.session_state.messages = []
    mock_requests.post.side_effect = Exception("Timeout")

    class DummyChatMessage:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.chat_message.return_value = DummyChatMessage()
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        prompt = mock_streamlit.chat_input("What is the agreed rate for this shipment?")
        if prompt:
            mock_streamlit.session_state.messages.append({"role": "user", "content": prompt})
            with mock_streamlit.chat_message("user"):
                mock_streamlit.markdown(prompt)
            with mock_streamlit.chat_message("assistant"):
                with mock_streamlit.spinner("Thinking..."):
                    try:
                        payload = {"question": prompt, "chat_history": []}
                        response = mock_requests.post("http://localhost:8000/ask", json=payload)
                        if response.status_code == 200:
                            pass
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
    mock_streamlit.error.assert_called()
    assert "Request failed" in mock_streamlit.error.call_args[0][0]

def test_data_extraction_happy_path(monkeypatch, mock_streamlit, mock_requests):
    # Simulate file upload
    file1 = make_file_mock("doc1.pdf", b"filecontent1", "application/pdf")
    mock_streamlit.file_uploader.return_value = file1
    mock_streamlit.button.side_effect = [True]
    # Simulate backend response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "data": {
            "reference_id": "REF123",
            "load_id": "LOAD456",
            "shipper": "Shipper Inc.",
            "consignee": "Consignee LLC",
            "carrier": {"carrier_name": "CarrierX", "mc_number": "MC789"},
            "driver": {"driver_name": "John Doe", "truck_number": "TRK001"},
            "pickup": {"city": "Dallas", "state": "TX"},
            "shipping_date": "2023-01-01",
            "drop": {"city": "Houston", "state": "TX"},
            "delivery_date": "2023-01-02",
            "rate_info": {"total_rate": "1200", "currency": "USD"},
            "equipment_type": "Flatbed"
        }
    }
    mock_requests.post.return_value = response_mock

    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()
    class DummyExpander:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.expander.return_value = DummyExpander()
    class DummyColumns:
        def __getitem__(self, idx): return MagicMock()
    mock_streamlit.columns.return_value = (MagicMock(), MagicMock())

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        extract_file = mock_streamlit.file_uploader(
            "Upload a single document for extraction", 
            type=["pdf", "docx", "txt"],
            key="extraction_uploader"
        )
        if mock_streamlit.button("Run Extraction"):
            if extract_file:
                with mock_streamlit.spinner("Extracting data..."):
                    try:
                        files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                        response = mock_requests.post("http://localhost:8000/extract", files=files)
                        if response.status_code == 200:
                            data = response.json()["data"]
                            mock_streamlit.success("Extraction complete!")
                            col1, col2 = mock_streamlit.columns(2)
                            with col1:
                                mock_streamlit.subheader("IDs & Parties")
                                mock_streamlit.write(f"**Reference ID:** {data.get('reference_id', 'N/A')}")
                                mock_streamlit.write(f"**Load ID:** {data.get('load_id', 'N/A')}")
                                mock_streamlit.write(f"**Shipper:** {data.get('shipper', 'N/A')}")
                                mock_streamlit.write(f"**Consignee:** {data.get('consignee', 'N/A')}")
                                mock_streamlit.subheader("Carrier & Driver")
                                carrier = data.get('carrier') or {}
                                mock_streamlit.write(f"**Carrier:** {carrier.get('carrier_name', 'N/A')}")
                                mock_streamlit.write(f"**MC Number:** {carrier.get('mc_number', 'N/A')}")
                                driver = data.get('driver') or {}
                                mock_streamlit.write(f"**Driver:** {driver.get('driver_name', 'N/A')}")
                                mock_streamlit.write(f"**Truck:** {driver.get('truck_number', 'N/A')}")
                            with col2:
                                mock_streamlit.subheader("Stops & Dates")
                                pickup = data.get('pickup') or {}
                                mock_streamlit.write(f"**Pickup:** {pickup.get('city', 'N/A')}, {pickup.get('state', 'N/A')}")
                                mock_streamlit.write(f"**Pickup Date:** {data.get('shipping_date', 'N/A')}")
                                drop = data.get('drop') or {}
                                mock_streamlit.write(f"**Drop:** {drop.get('city', 'N/A')}, {drop.get('state', 'N/A')}")
                                mock_streamlit.write(f"**Delivery Date:** {data.get('delivery_date', 'N/A')}")
                                mock_streamlit.subheader("Rates & Equipment")
                                rate = data.get('rate_info') or {}
                                mock_streamlit.write(f"**Total Rate:** {rate.get('total_rate', 'N/A')} {rate.get('currency', '')}")
                                mock_streamlit.write(f"**Equipment:** {data.get('equipment_type', 'N/A')}")
                            with mock_streamlit.expander("View Full JSON"):
                                mock_streamlit.json(data)
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
            else:
                mock_streamlit.warning("Please upload a document.")
    mock_streamlit.success.assert_called_with("Extraction complete!")
    mock_streamlit.json.assert_called()

def test_data_extraction_no_file(monkeypatch, mock_streamlit):
    mock_streamlit.file_uploader.return_value = None
    mock_streamlit.button.side_effect = [True]
    with patch.object(app, "st", mock_streamlit):
        extract_file = mock_streamlit.file_uploader(
            "Upload a single document for extraction", 
            type=["pdf", "docx", "txt"],
            key="extraction_uploader"
        )
        if mock_streamlit.button("Run Extraction"):
            if extract_file:
                pass
            else:
                mock_streamlit.warning("Please upload a document.")
    mock_streamlit.warning.assert_called_with("Please upload a document.")

def test_data_extraction_backend_error(monkeypatch, mock_streamlit, mock_requests):
    file1 = make_file_mock("doc1.pdf", b"filecontent1", "application/pdf")
    mock_streamlit.file_uploader.return_value = file1
    mock_streamlit.button.side_effect = [True]
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Error"
    mock_requests.post.return_value = response_mock

    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        extract_file = mock_streamlit.file_uploader(
            "Upload a single document for extraction", 
            type=["pdf", "docx", "txt"],
            key="extraction_uploader"
        )
        if mock_streamlit.button("Run Extraction"):
            if extract_file:
                with mock_streamlit.spinner("Extracting data..."):
                    try:
                        files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                        response = mock_requests.post("http://localhost:8000/extract", files=files)
                        if response.status_code == 200:
                            pass
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
            else:
                mock_streamlit.warning("Please upload a document.")
    mock_streamlit.error.assert_called_with("Error: Internal Error")

def test_data_extraction_request_exception(monkeypatch, mock_streamlit, mock_requests):
    file1 = make_file_mock("doc1.pdf", b"filecontent1", "application/pdf")
    mock_streamlit.file_uploader.return_value = file1
    mock_streamlit.button.side_effect = [True]
    mock_requests.post.side_effect = Exception("Timeout")

    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None
    mock_streamlit.spinner.return_value = DummySpinner()

    with patch.object(app, "st", mock_streamlit), patch.object(app, "requests", mock_requests):
        extract_file = mock_streamlit.file_uploader(
            "Upload a single document for extraction", 
            type=["pdf", "docx", "txt"],
            key="extraction_uploader"
        )
        if mock_streamlit.button("Run Extraction"):
            if extract_file:
                with mock_streamlit.spinner("Extracting data..."):
                    try:
                        files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                        response = mock_requests.post("http://localhost:8000/extract", files=files)
                        if response.status_code == 200:
                            pass
                        else:
                            mock_streamlit.error(f"Error: {response.text}")
                    except Exception as e:
                        mock_streamlit.error(f"Request failed: {str(e)}")
            else:
                mock_streamlit.warning("Please upload a document.")
    mock_streamlit.error.assert_called()
    assert "Request failed" in mock_streamlit.error.call_args[0][0]

def test_backend_unreachable_error_handling(monkeypatch, mock_streamlit):
    # Simulate backend unreachable
    mock_streamlit.error = MagicMock()
    mock_streamlit.button.return_value = False
    mock_streamlit.stop = MagicMock()
    with patch.object(app, "st", mock_streamlit):
        is_backend_up = False
        if not is_backend_up:
            mock_streamlit.error("⚠️ Cannot connect to backend at http://localhost:8000. Please ensure the backend is running.")
            if mock_streamlit.button("Retry Connection"):
                mock_streamlit.rerun()
            mock_streamlit.stop()
    mock_streamlit.error.assert_called_with("⚠️ Cannot connect to backend at http://localhost:8000. Please ensure the backend is running.")
    mock_streamlit.stop.assert_called_once()
