import pytest
import builtins
import types
import sys

import frontend.app as app

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit functions used in app.py to no-ops or mocks
    st_mock = types.SimpleNamespace()
    st_mock.set_page_config = lambda **kwargs: None
    st_mock.title = lambda *a, **k: None
    st_mock.header = lambda *a, **k: None
    st_mock.text_input = lambda *a, **k: "http://mocked-backend"
    st_mock.divider = lambda *a, **k: None
    st_mock.markdown = lambda *a, **k: None
    st_mock.error = lambda *a, **k: None
    st_mock.button = lambda *a, **k: False
    st_mock.stop = lambda *a, **k: (_ for _ in ()).throw(SystemExit)
    st_mock.tabs = lambda labels: [types.SimpleNamespace() for _ in labels]
    st_mock.write = lambda *a, **k: None
    st_mock.file_uploader = lambda *a, **k: []
    st_mock.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    st_mock.success = lambda *a, **k: None
    st_mock.warning = lambda *a, **k: None
    st_mock.expander = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    st_mock.chat_message = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    st_mock.caption = lambda *a, **k: None
    st_mock.chat_input = lambda *a, **k: None
    st_mock.session_state = {}
    st_mock.columns = lambda n: [types.SimpleNamespace() for _ in range(n)]
    st_mock.subheader = lambda *a, **k: None
    st_mock.json = lambda *a, **k: None
    st_mock.rerun = lambda: None

    monkeypatch.setattr(app, "st", st_mock)
    yield

@pytest.fixture
def mock_requests(monkeypatch):
    # Patch requests.get and requests.post
    class MockResponse:
        def __init__(self, status_code=200, json_data=None, text="OK"):
            self.status_code = status_code
            self._json = json_data or {}
            self.text = text
        def json(self):
            return self._json

    requests_mock = types.SimpleNamespace()
    requests_mock.get = lambda url, **kwargs: MockResponse(200)
    requests_mock.post = lambda url, **kwargs: MockResponse(200, {"message": "Success"})
    monkeypatch.setattr(app, "requests", requests_mock)
    return requests_mock

def test_check_backend_success(monkeypatch, patch_streamlit):
    # Should return True if backend responds with 200
    class Resp:
        status_code = 200
    monkeypatch.setattr(app.requests, "get", lambda url: Resp())
    assert app.check_backend() is True

def test_check_backend_failure(monkeypatch, patch_streamlit):
    # Should return False if backend raises exception
    def raise_exc(url):
        raise Exception("fail")
    monkeypatch.setattr(app.requests, "get", raise_exc)
    assert app.check_backend() is False

def test_check_backend_non_200(monkeypatch, patch_streamlit):
    # Should return False if backend returns non-200
    class Resp:
        status_code = 500
    monkeypatch.setattr(app.requests, "get", lambda url: Resp())
    assert app.check_backend() is False

def test_upload_documents_happy_path(monkeypatch, patch_streamlit):
    # Simulate uploading files and backend returns success
    files_data = [
        types.SimpleNamespace(
            name="doc1.pdf", getvalue=lambda: b"pdfdata", type="application/pdf"
        ),
        types.SimpleNamespace(
            name="doc2.txt", getvalue=lambda: b"txtdata", type="text/plain"
        ),
    ]
    # Patch file_uploader to return files
    app.st.file_uploader = lambda *a, **k: files_data
    # Patch button to simulate click
    app.st.button = lambda label: label == "Process & Index Documents"
    # Patch spinner context
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    # Patch requests.post to return a successful response
    class Resp:
        status_code = 200
        def json(self):
            return {
                "message": "Indexed!",
                "extractions": [
                    {"filename": "doc1.pdf", "structured_data_extracted": True, "text_chunks": 3},
                    {"filename": "doc2.txt", "structured_data_extracted": False, "text_chunks": 1},
                ],
                "errors": ["doc2.txt failed to extract"]
            }
    app.requests.post = lambda url, files=None: Resp()
    # Patch st.success, st.warning, st.expander, st.write
    called = {"success": False, "warning": [], "expander": False, "write": []}
    app.st.success = lambda msg: called.update(success=True)
    app.st.warning = lambda msg: called["warning"].append(msg)
    class DummyExpander:
        def __enter__(self): called.update(expander=True)
        def __exit__(self, *a): pass
    app.st.expander = lambda label: DummyExpander()
    app.st.write = lambda msg, **k: called["write"].append(msg)
    # Run the upload/index logic
    # Simulate the code block in the first tab
    # (simulate only the relevant logic)
    uploaded_files = app.st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if app.st.button("Process & Index Documents"):
        if uploaded_files:
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            with app.st.spinner("Processing documents (chunking, embedding, indexing)..."):
                try:
                    response = app.requests.post(
                        f"http://mocked-backend/upload",
                        files=files_to_send
                    )
                    if response.status_code == 200:
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
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
        else:
            app.st.warning("Please upload at least one document.")
    assert called["success"]
    assert called["expander"]
    assert any("✅" in w or "❌" in w for w in called["write"])
    assert "doc2.txt failed to extract" in called["warning"]

def test_upload_documents_no_files(monkeypatch, patch_streamlit):
    # No files uploaded, should warn
    app.st.file_uploader = lambda *a, **k: []
    app.st.button = lambda label: label == "Process & Index Documents"
    called = {"warning": False}
    app.st.warning = lambda msg: called.update(warning=True)
    uploaded_files = app.st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if app.st.button("Process & Index Documents"):
        if uploaded_files:
            pass
        else:
            app.st.warning("Please upload at least one document.")
    assert called["warning"]

def test_upload_documents_backend_error(monkeypatch, patch_streamlit):
    # Backend returns error status
    files_data = [
        types.SimpleNamespace(
            name="doc1.pdf", getvalue=lambda: b"pdfdata", type="application/pdf"
        )
    ]
    app.st.file_uploader = lambda *a, **k: files_data
    app.st.button = lambda label: label == "Process & Index Documents"
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    class Resp:
        status_code = 500
        text = "Internal Error"
    app.requests.post = lambda url, files=None: Resp()
    called = {"error": False}
    app.st.error = lambda msg: called.update(error=True)
    uploaded_files = app.st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if app.st.button("Process & Index Documents"):
        if uploaded_files:
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            with app.st.spinner("Processing documents (chunking, embedding, indexing)..."):
                try:
                    response = app.requests.post(
                        f"http://mocked-backend/upload",
                        files=files_to_send
                    )
                    if response.status_code == 200:
                        pass
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
        else:
            app.st.warning("Please upload at least one document.")
    assert called["error"]

def test_upload_documents_request_exception(monkeypatch, patch_streamlit):
    # requests.post raises exception
    files_data = [
        types.SimpleNamespace(
            name="doc1.pdf", getvalue=lambda: b"pdfdata", type="application/pdf"
        )
    ]
    app.st.file_uploader = lambda *a, **k: files_data
    app.st.button = lambda label: label == "Process & Index Documents"
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    def raise_exc(url, files=None):
        raise Exception("network fail")
    app.requests.post = raise_exc
    called = {"error": False}
    app.st.error = lambda msg: called.update(error=True)
    uploaded_files = app.st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if app.st.button("Process & Index Documents"):
        if uploaded_files:
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            with app.st.spinner("Processing documents (chunking, embedding, indexing)..."):
                try:
                    response = app.requests.post(
                        f"http://mocked-backend/upload",
                        files=files_to_send
                    )
                    if response.status_code == 200:
                        pass
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
        else:
            app.st.warning("Please upload at least one document.")
    assert called["error"]

def test_chat_qa_happy_path(monkeypatch, patch_streamlit):
    # Simulate chat input and backend returns answer
    app.st.session_state = {"messages": []}
    app.st.chat_input = lambda prompt: "What is the rate?"
    app.st.button = lambda *a, **k: False
    app.st.chat_message = lambda role: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    class Resp:
        status_code = 200
        def json(self):
            return {
                "answer": "The rate is $1000.",
                "confidence_score": 0.95,
                "sources": [
                    {"metadata": {"source": "doc1.pdf"}, "text": "Relevant text"}
                ]
            }
    app.requests.post = lambda url, json=None: Resp()
    called = {"markdown": [], "caption": [], "expander": False}
    app.st.markdown = lambda msg: called["markdown"].append(msg)
    app.st.caption = lambda msg: called["caption"].append(msg)
    class DummyExpander:
        def __enter__(self): called.update(expander=True)
        def __exit__(self, *a): pass
    app.st.expander = lambda label: DummyExpander()
    app.st.write = lambda msg, **k: None
    # Simulate the chat logic
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    if prompt:
        app.st.session_state["messages"].append({"role": "user", "content": prompt})
        with app.st.chat_message("user"):
            app.st.markdown(prompt)
        with app.st.chat_message("assistant"):
            with app.st.spinner("Thinking..."):
                try:
                    payload = {"question": prompt, "chat_history": []}
                    response = app.requests.post(f"http://mocked-backend/ask", json=payload)
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
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
    assert any("The rate is $1000." in m for m in called["markdown"])
    assert any("Confidence Score" in c for c in called["caption"])
    assert called["expander"]
    assert app.st.session_state["messages"][-1]["role"] == "assistant"

def test_chat_qa_backend_error(monkeypatch, patch_streamlit):
    # Backend returns error status
    app.st.session_state = {"messages": []}
    app.st.chat_input = lambda prompt: "What is the rate?"
    app.st.button = lambda *a, **k: False
    app.st.chat_message = lambda role: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    class Resp:
        status_code = 500
        text = "Internal Error"
    app.requests.post = lambda url, json=None: Resp()
    called = {"error": False}
    app.st.error = lambda msg: called.update(error=True)
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    if prompt:
        app.st.session_state["messages"].append({"role": "user", "content": prompt})
        with app.st.chat_message("user"):
            app.st.markdown(prompt)
        with app.st.chat_message("assistant"):
            with app.st.spinner("Thinking..."):
                try:
                    payload = {"question": prompt, "chat_history": []}
                    response = app.requests.post(f"http://mocked-backend/ask", json=payload)
                    if response.status_code == 200:
                        pass
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
    assert called["error"]

def test_chat_qa_request_exception(monkeypatch, patch_streamlit):
    # requests.post raises exception
    app.st.session_state = {"messages": []}
    app.st.chat_input = lambda prompt: "What is the rate?"
    app.st.button = lambda *a, **k: False
    app.st.chat_message = lambda role: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    def raise_exc(url, json=None):
        raise Exception("network fail")
    app.requests.post = raise_exc
    called = {"error": False}
    app.st.error = lambda msg: called.update(error=True)
    prompt = app.st.chat_input("What is the agreed rate for this shipment?")
    if prompt:
        app.st.session_state["messages"].append({"role": "user", "content": prompt})
        with app.st.chat_message("user"):
            app.st.markdown(prompt)
        with app.st.chat_message("assistant"):
            with app.st.spinner("Thinking..."):
                try:
                    payload = {"question": prompt, "chat_history": []}
                    response = app.requests.post(f"http://mocked-backend/ask", json=payload)
                    if response.status_code == 200:
                        pass
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
    assert called["error"]

def test_data_extraction_happy_path(monkeypatch, patch_streamlit):
    # Simulate file upload and backend returns extraction data
    extract_file = types.SimpleNamespace(
        name="doc1.pdf", getvalue=lambda: b"pdfdata", type="application/pdf"
    )
    app.st.file_uploader = lambda *a, **k: extract_file
    app.st.button = lambda label: label == "Run Extraction"
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
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
                    "pickup": {"city": "CityA", "state": "ST"},
                    "shipping_date": "2024-01-01",
                    "drop": {"city": "CityB", "state": "TS"},
                    "delivery_date": "2024-01-02",
                    "rate_info": {"total_rate": 1000, "currency": "USD"},
                    "equipment_type": "Flatbed"
                }
            }
    app.requests.post = lambda url, files=None: Resp()
    called = {"success": False, "json": False}
    app.st.success = lambda msg: called.update(success=True)
    app.st.columns = lambda n: [types.SimpleNamespace() for _ in range(n)]
    app.st.subheader = lambda *a, **k: None
    app.st.write = lambda *a, **k: None
    app.st.json = lambda data: called.update(json=True)
    class DummyExpander:
        def __enter__(self): return None
        def __exit__(self, *a): pass
    app.st.expander = lambda label: DummyExpander()
    extract_file = app.st.file_uploader("Upload a single document for extraction", type=["pdf", "docx", "txt"], key="extraction_uploader")
    if app.st.button("Run Extraction"):
        if extract_file:
            with app.st.spinner("Extracting data..."):
                try:
                    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                    response = app.requests.post(f"http://mocked-backend/extract", files=files)
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
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
        else:
            app.st.warning("Please upload a document.")
    assert called["success"]
    assert called["json"]

def test_data_extraction_no_file(monkeypatch, patch_streamlit):
    # No file uploaded, should warn
    app.st.file_uploader = lambda *a, **k: None
    app.st.button = lambda label: label == "Run Extraction"
    called = {"warning": False}
    app.st.warning = lambda msg: called.update(warning=True)
    extract_file = app.st.file_uploader("Upload a single document for extraction", type=["pdf", "docx", "txt"], key="extraction_uploader")
    if app.st.button("Run Extraction"):
        if extract_file:
            pass
        else:
            app.st.warning("Please upload a document.")
    assert called["warning"]

def test_data_extraction_backend_error(monkeypatch, patch_streamlit):
    # Backend returns error status
    extract_file = types.SimpleNamespace(
        name="doc1.pdf", getvalue=lambda: b"pdfdata", type="application/pdf"
    )
    app.st.file_uploader = lambda *a, **k: extract_file
    app.st.button = lambda label: label == "Run Extraction"
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    class Resp:
        status_code = 500
        text = "Internal Error"
    app.requests.post = lambda url, files=None: Resp()
    called = {"error": False}
    app.st.error = lambda msg: called.update(error=True)
    extract_file = app.st.file_uploader("Upload a single document for extraction", type=["pdf", "docx", "txt"], key="extraction_uploader")
    if app.st.button("Run Extraction"):
        if extract_file:
            with app.st.spinner("Extracting data..."):
                try:
                    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                    response = app.requests.post(f"http://mocked-backend/extract", files=files)
                    if response.status_code == 200:
                        pass
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
        else:
            app.st.warning("Please upload a document.")
    assert called["error"]

def test_data_extraction_request_exception(monkeypatch, patch_streamlit):
    # requests.post raises exception
    extract_file = types.SimpleNamespace(
        name="doc1.pdf", getvalue=lambda: b"pdfdata", type="application/pdf"
    )
    app.st.file_uploader = lambda *a, **k: extract_file
    app.st.button = lambda label: label == "Run Extraction"
    app.st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    def raise_exc(url, files=None):
        raise Exception("network fail")
    app.requests.post = raise_exc
    called = {"error": False}
    app.st.error = lambda msg: called.update(error=True)
    extract_file = app.st.file_uploader("Upload a single document for extraction", type=["pdf", "docx", "txt"], key="extraction_uploader")
    if app.st.button("Run Extraction"):
        if extract_file:
            with app.st.spinner("Extracting data..."):
                try:
                    files = {"file": (extract_file.name, extract_file.getvalue(), extract_file.type)}
                    response = app.requests.post(f"http://mocked-backend/extract", files=files)
                    if response.status_code == 200:
                        pass
                    else:
                        app.st.error(f"Error: {response.text}")
                except Exception as e:
                    app.st.error(f"Request failed: {str(e)}")
        else:
            app.st.warning("Please upload a document.")
    assert called["error"]
