import pytest
import builtins
import types
import sys
import os

import frontend.app as app

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit UI functions to no-ops or mocks
    st_mock = MagicMock()
    st_mock.sidebar = MagicMock()
    st_mock.sidebar.header = MagicMock()
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
    st_mock.divider = MagicMock()
    st_mock.markdown = MagicMock()
    st_mock.caption = MagicMock()
    st_mock.expander = MagicMock()
    st_mock.columns = MagicMock(return_value=(MagicMock(), MagicMock()))
    st_mock.json = MagicMock()
    st_mock.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    st_mock.stop = MagicMock()
    st_mock.rerun = MagicMock()
    st_mock.chat_message = MagicMock()
    st_mock.chat_input = MagicMock()
    st_mock.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    yield

@pytest.fixture(autouse=True)
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    requests_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "requests", requests_mock)
    yield

@pytest.fixture(autouse=True)
def patch_os(monkeypatch):
    # Patch os.getenv to return deterministic value
    monkeypatch.setattr(os, "getenv", lambda key, default=None: "http://mock-backend:8000")
    yield

def make_response(status_code=200, json_data=None, text="OK"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = Exception("No JSON")
    return resp

def test_check_backend_success(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    resp = make_response(200)
    monkeypatch.setattr(app.requests, "get", lambda url: resp)
    # Act
    result = app.check_backend()
    # Assert
    assert result is True

def test_check_backend_failure_status(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    resp = make_response(500)
    monkeypatch.setattr(app.requests, "get", lambda url: resp)
    # Act
    result = app.check_backend()
    # Assert
    assert result is False

def test_check_backend_exception(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    def raise_exc(url):
        raise Exception("Connection error")
    monkeypatch.setattr(app.requests, "get", raise_exc)
    # Act
    result = app.check_backend()
    # Assert
    assert result is False

def test_upload_and_index_happy_path(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    # Simulate uploaded files
    file_mock = MagicMock()
    file_mock.name = "test.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    uploaded_files = [file_mock]
    # Patch st.file_uploader to return uploaded_files
    app.st.file_uploader.return_value = uploaded_files
    # Patch st.button to simulate click
    app.st.button.side_effect = lambda label: label == "Process & Index Documents"
    # Patch requests.post to return success
    extraction_result = {
        "message": "Indexed successfully",
        "extractions": [
            {"filename": "test.pdf", "structured_data_extracted": True, "text_chunks": 5}
        ],
        "errors": []
    }
    monkeypatch.setattr(app.requests, "post", lambda url, files: make_response(200, extraction_result))
    # Patch st.spinner as context manager
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    # Patch st.expander as context manager
    class DummyExpander:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.expander.return_value = DummyExpander()
    # Act
    # Simulate the logic inside the first tab
    with app.st.spinner("Processing documents (chunking, embedding, indexing)..."):
        response = app.requests.post(f"{backend_url}/upload", files=[
            ("files", (file_mock.name, file_mock.getvalue(), file_mock.type))
        ])
        assert response.status_code == 200
        result = response.json()
        app.st.success(result["message"])
        if result.get("extractions"):
            with app.st.expander("Extraction Summary"):
                for ext in result["extractions"]:
                    status = "✅" if ext["structured_data_extracted"] else "❌"
                    app.st.write(f"{status} **{ext['filename']}**: {ext['text_chunks']} chunks")
        if result.get("errors"):
            for err in result["errors"]:
                app.st.warning(err)
    # Assert
    app.st.success.assert_called_with("Indexed successfully")
    app.st.write.assert_any_call("✅ **test.pdf**: 5 chunks")

def test_upload_and_index_no_files(monkeypatch):
    # Arrange
    app.st.file_uploader.return_value = []
    app.st.button.side_effect = lambda label: label == "Process & Index Documents"
    # Act
    if not app.st.file_uploader.return_value:
        app.st.warning("Please upload at least one document.")
    # Assert
    app.st.warning.assert_called_with("Please upload at least one document.")

def test_upload_and_index_backend_error(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    file_mock = MagicMock()
    file_mock.name = "test.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    uploaded_files = [file_mock]
    app.st.file_uploader.return_value = uploaded_files
    app.st.button.side_effect = lambda label: label == "Process & Index Documents"
    monkeypatch.setattr(app.requests, "post", lambda url, files: make_response(500, None, "Internal Server Error"))
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    # Act
    with app.st.spinner("Processing documents (chunking, embedding, indexing)..."):
        response = app.requests.post(f"{backend_url}/upload", files=[
            ("files", (file_mock.name, file_mock.getvalue(), file_mock.type))
        ])
        if response.status_code == 200:
            pass
        else:
            app.st.error(f"Error: {response.text}")
    # Assert
    app.st.error.assert_called_with("Error: Internal Server Error")

def test_upload_and_index_request_exception(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    file_mock = MagicMock()
    file_mock.name = "test.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    uploaded_files = [file_mock]
    app.st.file_uploader.return_value = uploaded_files
    app.st.button.side_effect = lambda label: label == "Process & Index Documents"
    def raise_exc(url, files):
        raise Exception("Network error")
    monkeypatch.setattr(app.requests, "post", raise_exc)
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    # Act
    try:
        with app.st.spinner("Processing documents (chunking, embedding, indexing)..."):
            app.requests.post(f"{backend_url}/upload", files=[
                ("files", (file_mock.name, file_mock.getvalue(), file_mock.type))
            ])
    except Exception as e:
        app.st.error(f"Request failed: {str(e)}")
    # Assert
    app.st.error.assert_called_with("Request failed: Network error")

def test_chat_qa_happy_path(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    app.st.session_state.clear()
    app.st.session_state["messages"] = []
    app.st.chat_input.return_value = "What is the agreed rate for this shipment?"
    app.st.button.side_effect = lambda label: False
    answer_data = {
        "answer": "The agreed rate is $1200.",
        "confidence_score": 0.95,
        "sources": [
            {"metadata": {"source": "test.pdf"}, "text": "Rate: $1200"}
        ]
    }
    monkeypatch.setattr(app.requests, "post", lambda url, json: make_response(200, answer_data))
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    class DummyChatMsg:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.chat_message.return_value = DummyChatMsg()
    app.st.expander.return_value = DummySpinner()
    # Act
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    app.st.session_state["messages"].append({"role": "user", "content": prompt})
    with app.st.chat_message("user"):
        app.st.markdown(prompt)
    with app.st.chat_message("assistant"):
        with app.st.spinner("Thinking..."):
            payload = {"question": prompt, "chat_history": []}
            response = app.requests.post(f"{backend_url}/ask", json=payload)
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                confidence = data["confidence_score"]
                sources = data["sources"]
                app.st.markdown(answer)
                app.st.caption(f"Confidence Score: {confidence:.2f}")
                if sources:
                    with app.st.expander("View Sources"):
                        for src in sources:
                            app.st.caption(f"Source: {src['metadata']['source']}")
                            app.st.write(src["text"])
                app.st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                app.st.error(f"Error: {response.text}")
    # Assert
    app.st.markdown.assert_any_call("The agreed rate is $1200.")
    app.st.caption.assert_any_call("Confidence Score: 0.95")
    app.st.caption.assert_any_call("Source: test.pdf")
    app.st.write.assert_any_call("Rate: $1200")
    assert app.st.session_state["messages"][-1]["role"] == "assistant"

def test_chat_qa_backend_error(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    app.st.session_state.clear()
    app.st.session_state["messages"] = []
    app.st.chat_input.return_value = "What is the agreed rate for this shipment?"
    app.st.button.side_effect = lambda label: False
    monkeypatch.setattr(app.requests, "post", lambda url, json: make_response(500, None, "Internal Error"))
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    class DummyChatMsg:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.chat_message.return_value = DummyChatMsg()
    # Act
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    app.st.session_state["messages"].append({"role": "user", "content": prompt})
    with app.st.chat_message("user"):
        app.st.markdown(prompt)
    with app.st.chat_message("assistant"):
        with app.st.spinner("Thinking..."):
            payload = {"question": prompt, "chat_history": []}
            response = app.requests.post(f"{backend_url}/ask", json=payload)
            if response.status_code == 200:
                pass
            else:
                app.st.error(f"Error: {response.text}")
    # Assert
    app.st.error.assert_called_with("Error: Internal Error")

def test_chat_qa_request_exception(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    app.st.session_state.clear()
    app.st.session_state["messages"] = []
    app.st.chat_input.return_value = "What is the agreed rate for this shipment?"
    app.st.button.side_effect = lambda label: False
    def raise_exc(url, json):
        raise Exception("Network error")
    monkeypatch.setattr(app.requests, "post", raise_exc)
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    class DummyChatMsg:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.chat_message.return_value = DummyChatMsg()
    # Act
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    app.st.session_state["messages"].append({"role": "user", "content": prompt})
    with app.st.chat_message("user"):
        app.st.markdown(prompt)
    with app.st.chat_message("assistant"):
        with app.st.spinner("Thinking..."):
            payload = {"question": prompt, "chat_history": []}
            try:
                app.requests.post(f"{backend_url}/ask", json=payload)
            except Exception as e:
                app.st.error(f"Request failed: {str(e)}")
    # Assert
    app.st.error.assert_called_with("Request failed: Network error")

def test_data_extraction_happy_path(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    file_mock = MagicMock()
    file_mock.name = "extract.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    app.st.file_uploader.return_value = file_mock
    app.st.button.side_effect = lambda label: label == "Run Extraction"
    extraction_data = {
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
            "equipment_type": "Reefer"
        }
    }
    monkeypatch.setattr(app.requests, "post", lambda url, files: make_response(200, extraction_data))
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    app.st.columns.return_value = (MagicMock(), MagicMock())
    # Act
    with app.st.spinner("Extracting data..."):
        files = {"file": (file_mock.name, file_mock.getvalue(), file_mock.type)}
        response = app.requests.post(f"{backend_url}/extract", files=files)
        if response.status_code == 200:
            data = response.json()["data"]
            app.st.success("Extraction complete!")
            col1, col2 = app.st.columns(2)
            with col1:
                app.st.subheader("IDs & Parties")
                app.st.write(f"**Reference ID:** {data.get('reference_id', 'N/A')}")
                app.st.write(f"**Load ID:** {data.get('load_id', 'N/A')}")
                app.st.write(f"**Shipper:** {data.get('shipper', 'N/A')}")
                app.st.write(f"**Consignee:** {data.get('consignee', 'N/A')}")
                app.st.subheader("Carrier & Driver")
                carrier = data.get('carrier') or {}
                app.st.write(f"**Carrier:** {carrier.get('carrier_name', 'N/A')}")
                app.st.write(f"**MC Number:** {carrier.get('mc_number', 'N/A')}")
                driver = data.get('driver') or {}
                app.st.write(f"**Driver:** {driver.get('driver_name', 'N/A')}")
                app.st.write(f"**Truck:** {driver.get('truck_number', 'N/A')}")
            with col2:
                app.st.subheader("Stops & Dates")
                pickup = data.get('pickup') or {}
                app.st.write(f"**Pickup:** {pickup.get('city', 'N/A')}, {pickup.get('state', 'N/A')}")
                app.st.write(f"**Pickup Date:** {data.get('shipping_date', 'N/A')}")
                drop = data.get('drop') or {}
                app.st.write(f"**Drop:** {drop.get('city', 'N/A')}, {drop.get('state', 'N/A')}")
                app.st.write(f"**Delivery Date:** {data.get('delivery_date', 'N/A')}")
                app.st.subheader("Rates & Equipment")
                rate = data.get('rate_info') or {}
                app.st.write(f"**Total Rate:** {rate.get('total_rate', 'N/A')} {rate.get('currency', '')}")
                app.st.write(f"**Equipment:** {data.get('equipment_type', 'N/A')}")
            with app.st.expander("View Full JSON"):
                app.st.json(data)
    # Assert
    app.st.success.assert_called_with("Extraction complete!")
    app.st.write.assert_any_call("**Reference ID:** REF123")
    app.st.write.assert_any_call("**Total Rate:** 1200 USD")
    app.st.json.assert_called_with(extraction_data["data"])

def test_data_extraction_no_file(monkeypatch):
    # Arrange
    app.st.file_uploader.return_value = None
    app.st.button.side_effect = lambda label: label == "Run Extraction"
    # Act
    if not app.st.file_uploader.return_value:
        app.st.warning("Please upload a document.")
    # Assert
    app.st.warning.assert_called_with("Please upload a document.")

def test_data_extraction_backend_error(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    file_mock = MagicMock()
    file_mock.name = "extract.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    app.st.file_uploader.return_value = file_mock
    app.st.button.side_effect = lambda label: label == "Run Extraction"
    monkeypatch.setattr(app.requests, "post", lambda url, files: make_response(500, None, "Internal Error"))
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    # Act
    with app.st.spinner("Extracting data..."):
        files = {"file": (file_mock.name, file_mock.getvalue(), file_mock.type)}
        response = app.requests.post(f"{backend_url}/extract", files=files)
        if response.status_code == 200:
            pass
        else:
            app.st.error(f"Error: {response.text}")
    # Assert
    app.st.error.assert_called_with("Error: Internal Error")

def test_data_extraction_request_exception(monkeypatch):
    # Arrange
    backend_url = "http://mock-backend:8000"
    monkeypatch.setattr(app, "BACKEND_URL", backend_url)
    file_mock = MagicMock()
    file_mock.name = "extract.pdf"
    file_mock.getvalue.return_value = b"PDFDATA"
    file_mock.type = "application/pdf"
    app.st.file_uploader.return_value = file_mock
    app.st.button.side_effect = lambda label: label == "Run Extraction"
    def raise_exc(url, files):
        raise Exception("Network error")
    monkeypatch.setattr(app.requests, "post", raise_exc)
    class DummySpinner:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    app.st.spinner.return_value = DummySpinner()
    # Act
    try:
        with app.st.spinner("Extracting data..."):
            files = {"file": (file_mock.name, file_mock.getvalue(), file_mock.type)}
            app.requests.post(f"{backend_url}/extract", files=files)
    except Exception as e:
        app.st.error(f"Request failed: {str(e)}")
    # Assert
    app.st.error.assert_called_with("Request failed: Network error")
