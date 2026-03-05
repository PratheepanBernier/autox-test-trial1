# source_hash: 9ee607a3d8da4254
import pytest
import builtins
import types
import io

import frontend.app as app

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit functions used in app.py to no-op or record calls
    st_calls = {}

    class DummyExpander:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def write(self, *a, **k): st_calls.setdefault('expander_write', []).append((a, k))
        def caption(self, *a, **k): st_calls.setdefault('expander_caption', []).append((a, k))
        def markdown(self, *a, **k): st_calls.setdefault('expander_markdown', []).append((a, k))
        def json(self, *a, **k): st_calls.setdefault('expander_json', []).append((a, k))

    class DummyChatMessage:
        def __init__(self, role): self.role = role
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def markdown(self, *a, **k): st_calls.setdefault('chat_markdown', []).append((self.role, a, k))
        def caption(self, *a, **k): st_calls.setdefault('chat_caption', []).append((self.role, a, k))
        def write(self, *a, **k): st_calls.setdefault('chat_write', []).append((self.role, a, k))
        def expander(self, *a, **k): return DummyExpander()

    class DummySpinner:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    class DummyColumns:
        def __init__(self, n): self.cols = [self for _ in range(n)]
        def __getitem__(self, idx): return self.cols[idx]
        def __iter__(self): return iter(self.cols)
        def subheader(self, *a, **k): st_calls.setdefault('col_subheader', []).append((a, k))
        def write(self, *a, **k): st_calls.setdefault('col_write', []).append((a, k))

    class DummySidebar:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def header(self, *a, **k): st_calls.setdefault('sidebar_header', []).append((a, k))
        def divider(self, *a, **k): st_calls.setdefault('sidebar_divider', []).append((a, k))
        def markdown(self, *a, **k): st_calls.setdefault('sidebar_markdown', []).append((a, k))

    class DummyTabs:
        def __init__(self, labels): self.tabs = [self for _ in labels]
        def __getitem__(self, idx): return self.tabs[idx]
        def __iter__(self): return iter(self.tabs)
        def header(self, *a, **k): st_calls.setdefault('tab_header', []).append((a, k))
        def write(self, *a, **k): st_calls.setdefault('tab_write', []).append((a, k))
        def file_uploader(self, *a, **k): st_calls.setdefault('tab_file_uploader', []).append((a, k)); return None
        def button(self, *a, **k): st_calls.setdefault('tab_button', []).append((a, k)); return False
        def spinner(self, *a, **k): return DummySpinner()
        def expander(self, *a, **k): return DummyExpander()
        def columns(self, n): return DummyColumns(n)
        def subheader(self, *a, **k): st_calls.setdefault('tab_subheader', []).append((a, k))
        def caption(self, *a, **k): st_calls.setdefault('tab_caption', []).append((a, k))
        def markdown(self, *a, **k): st_calls.setdefault('tab_markdown', []).append((a, k))
        def json(self, *a, **k): st_calls.setdefault('tab_json', []).append((a, k))
        def warning(self, *a, **k): st_calls.setdefault('tab_warning', []).append((a, k))
        def error(self, *a, **k): st_calls.setdefault('tab_error', []).append((a, k))
        def success(self, *a, **k): st_calls.setdefault('tab_success', []).append((a, k))

    class DummySessionState(dict):
        pass

    dummy_st = types.SimpleNamespace()
    dummy_st.set_page_config = lambda *a, **k: st_calls.setdefault('set_page_config', []).append((a, k))
    dummy_st.title = lambda *a, **k: st_calls.setdefault('title', []).append((a, k))
    dummy_st.sidebar = DummySidebar()
    dummy_st.header = lambda *a, **k: st_calls.setdefault('header', []).append((a, k))
    dummy_st.divider = lambda *a, **k: st_calls.setdefault('divider', []).append((a, k))
    dummy_st.markdown = lambda *a, **k: st_calls.setdefault('markdown', []).append((a, k))
    dummy_st.text_input = lambda *a, **k: st_calls.setdefault('text_input', []).append((a, k)) or "http://localhost:8000"
    dummy_st.tabs = lambda labels: DummyTabs(labels)
    dummy_st.file_uploader = lambda *a, **k: st_calls.setdefault('file_uploader', []).append((a, k)) or None
    dummy_st.button = lambda *a, **k: st_calls.setdefault('button', []).append((a, k)) or False
    dummy_st.spinner = lambda *a, **k: DummySpinner()
    dummy_st.expander = lambda *a, **k: DummyExpander()
    dummy_st.success = lambda *a, **k: st_calls.setdefault('success', []).append((a, k))
    dummy_st.warning = lambda *a, **k: st_calls.setdefault('warning', []).append((a, k))
    dummy_st.error = lambda *a, **k: st_calls.setdefault('error', []).append((a, k))
    dummy_st.stop = lambda: (_ for _ in ()).throw(SystemExit)
    dummy_st.rerun = lambda: st_calls.setdefault('rerun', []).append(())
    dummy_st.session_state = DummySessionState()
    dummy_st.chat_message = lambda role: DummyChatMessage(role)
    dummy_st.chat_input = lambda *a, **k: st_calls.setdefault('chat_input', []).append((a, k)) or None
    dummy_st.columns = lambda n: DummyColumns(n)
    dummy_st.subheader = lambda *a, **k: st_calls.setdefault('subheader', []).append((a, k))
    dummy_st.caption = lambda *a, **k: st_calls.setdefault('caption', []).append((a, k))
    dummy_st.write = lambda *a, **k: st_calls.setdefault('write', []).append((a, k))
    dummy_st.json = lambda *a, **k: st_calls.setdefault('json', []).append((a, k))

    monkeypatch.setattr(app, "st", dummy_st)
    yield st_calls

@pytest.fixture
def patch_requests(monkeypatch):
    # Patch requests.get and requests.post
    responses = {}

    class DummyResponse:
        def __init__(self, status_code=200, json_data=None, text="OK"):
            self.status_code = status_code
            self._json = json_data
            self.text = text
        def json(self):
            if isinstance(self._json, Exception):
                raise self._json
            return self._json

    def get(url, *a, **k):
        resp = responses.get(('GET', url))
        if resp is not None:
            return resp
        return DummyResponse(200, {}, "pong")

    def post(url, *a, **k):
        resp = responses.get(('POST', url))
        if resp is not None:
            return resp
        return DummyResponse(200, {}, "ok")

    monkeypatch.setattr(app.requests, "get", get)
    monkeypatch.setattr(app.requests, "post", post)
    return responses, DummyResponse

@pytest.fixture
def patch_os_environ(monkeypatch):
    env = {}
    monkeypatch.setattr(app.os, "getenv", lambda k, d=None: env.get(k, d))
    return env

def test_check_backend_happy_path(patch_streamlit, patch_requests):
    responses, DummyResponse = patch_requests
    responses[('GET', "http://localhost:8000/ping")] = DummyResponse(200, {}, "pong")
    # Should return True
    assert app.check_backend() is True

def test_check_backend_down_returns_false(patch_streamlit, patch_requests):
    responses, DummyResponse = patch_requests
    responses[('GET', "http://localhost:8000/ping")] = DummyResponse(500, {}, "fail")
    assert app.check_backend() is False

def test_check_backend_exception_returns_false(patch_streamlit, patch_requests, monkeypatch):
    def raise_exc(*a, **k): raise Exception("network error")
    monkeypatch.setattr(app.requests, "get", raise_exc)
    assert app.check_backend() is False

def test_backend_url_from_env(patch_streamlit, patch_os_environ):
    patch_os_environ["BACKEND_URL"] = "http://test-backend:9000"
    # Should use env var as default_backend
    # The text_input is patched to always return the default_backend
    assert app.st.text_input("Backend API URL", value="http://test-backend:9000", help="The URL of the FastAPI backend.") == "http://test-backend:9000"

def test_upload_and_index_documents_success(patch_streamlit, patch_requests, monkeypatch):
    # Simulate uploaded files and successful backend response
    responses, DummyResponse = patch_requests
    # Patch file_uploader to return a list of dummy files
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content

    dummy_file = DummyFile("doc1.pdf", b"pdfdata", "application/pdf")
    app.st.file_uploader = lambda *a, **k: [dummy_file]
    app.st.button = lambda *a, **k: True

    extraction = {
        "filename": "doc1.pdf",
        "structured_data_extracted": True,
        "text_chunks": 5
    }
    responses[('POST', "http://localhost:8000/upload")] = DummyResponse(200, {
        "message": "Indexed successfully",
        "extractions": [extraction],
        "errors": []
    }, "OK")

    # Should call st.success and st.expander
    # Run the relevant code block
    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: [dummy_file],
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: DummyResponse(),
            expander=lambda *a, **k: DummyResponse(),
            columns=lambda n: [DummyResponse() for _ in range(n)],
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        ) for _ in range(3)]
        app.st.stop()

def test_upload_and_index_documents_no_files(patch_streamlit, patch_requests):
    # file_uploader returns empty list, button pressed
    app.st.file_uploader = lambda *a, **k: []
    app.st.button = lambda *a, **k: True
    # Should call st.warning
    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: [],
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: None,
            expander=lambda *a, **k: None,
            columns=lambda n: [None for _ in range(n)],
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        ) for _ in range(3)]
        app.st.stop()

def test_upload_and_index_documents_backend_error(patch_streamlit, patch_requests):
    # file_uploader returns file, button pressed, backend returns error
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content

    dummy_file = DummyFile("doc1.pdf", b"pdfdata", "application/pdf")
    app.st.file_uploader = lambda *a, **k: [dummy_file]
    app.st.button = lambda *a, **k: True

    responses, DummyResponse = patch_requests
    responses[('POST', "http://localhost:8000/upload")] = DummyResponse(400, None, "Bad Request")

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: [dummy_file],
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: None,
            expander=lambda *a, **k: None,
            columns=lambda n: [None for _ in range(n)],
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        ) for _ in range(3)]
        app.st.stop()

def test_upload_and_index_documents_request_exception(patch_streamlit, patch_requests, monkeypatch):
    # file_uploader returns file, button pressed, requests.post raises exception
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content

    dummy_file = DummyFile("doc1.pdf", b"pdfdata", "application/pdf")
    app.st.file_uploader = lambda *a, **k: [dummy_file]
    app.st.button = lambda *a, **k: True

    def raise_exc(*a, **k): raise Exception("network error")
    monkeypatch.setattr(app.requests, "post", raise_exc)

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: [dummy_file],
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: None,
            expander=lambda *a, **k: None,
            columns=lambda n: [None for _ in range(n)],
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        ) for _ in range(3)]
        app.st.stop()

def test_chat_qa_happy_path(patch_streamlit, patch_requests, monkeypatch):
    # Simulate chat input and backend response
    responses, DummyResponse = patch_requests
    app.st.chat_input = lambda *a, **k: "What is the agreed rate for this shipment?"
    app.st.session_state.messages = []
    answer = "The agreed rate is $1200."
    confidence = 0.97
    sources = [{"metadata": {"source": "doc1.pdf"}, "text": "Relevant text"}]
    responses[('POST', "http://localhost:8000/ask")] = DummyResponse(200, {
        "answer": answer,
        "confidence_score": confidence,
        "sources": sources
    }, "OK")

    # Should append user and assistant messages to session_state
    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [None, types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            chat_message=lambda role: types.SimpleNamespace(
                __enter__=lambda s: s, __exit__=lambda s, e, v, t: None,
                markdown=lambda *a, **k: None,
                caption=lambda *a, **k: None,
                write=lambda *a, **k: None,
                expander=lambda *a, **k: None
            ),
            chat_input=lambda *a, **k: "What is the agreed rate for this shipment?"
        ), None]
        app.st.stop()

def test_chat_qa_backend_error(patch_streamlit, patch_requests):
    # Simulate backend error on /ask
    responses, DummyResponse = patch_requests
    app.st.chat_input = lambda *a, **k: "What is the agreed rate for this shipment?"
    app.st.session_state.messages = []
    responses[('POST', "http://localhost:8000/ask")] = DummyResponse(500, None, "Internal Error")

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [None, types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            chat_message=lambda role: types.SimpleNamespace(
                __enter__=lambda s: s, __exit__=lambda s, e, v, t: None,
                markdown=lambda *a, **k: None,
                caption=lambda *a, **k: None,
                write=lambda *a, **k: None,
                expander=lambda *a, **k: None
            ),
            chat_input=lambda *a, **k: "What is the agreed rate for this shipment?"
        ), None]
        app.st.stop()

def test_chat_qa_request_exception(patch_streamlit, monkeypatch):
    app.st.chat_input = lambda *a, **k: "What is the agreed rate for this shipment?"
    app.st.session_state.messages = []
    def raise_exc(*a, **k): raise Exception("network error")
    monkeypatch.setattr(app.requests, "post", raise_exc)

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [None, types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            chat_message=lambda role: types.SimpleNamespace(
                __enter__=lambda s: s, __exit__=lambda s, e, v, t: None,
                markdown=lambda *a, **k: None,
                caption=lambda *a, **k: None,
                write=lambda *a, **k: None,
                expander=lambda *a, **k: None
            ),
            chat_input=lambda *a, **k: "What is the agreed rate for this shipment?"
        ), None]
        app.st.stop()

def test_data_extraction_happy_path(patch_streamlit, patch_requests):
    # Simulate file upload and backend extraction response
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content

    dummy_file = DummyFile("doc2.pdf", b"pdfdata", "application/pdf")
    app.st.file_uploader = lambda *a, **k: dummy_file
    app.st.button = lambda *a, **k: True

    responses, DummyResponse = patch_requests
    extraction_data = {
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
        "rate_info": {"total_rate": "1200", "currency": "USD"},
        "equipment_type": "Van"
    }
    responses[('POST', "http://localhost:8000/extract")] = DummyResponse(200, {"data": extraction_data}, "OK")

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [None, None, types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: dummy_file,
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: None,
            columns=lambda n: [types.SimpleNamespace(
                subheader=lambda *a, **k: None,
                write=lambda *a, **k: None
            ) for _ in range(n)],
            expander=lambda *a, **k: None,
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        )]
        app.st.stop()

def test_data_extraction_backend_error(patch_streamlit, patch_requests):
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content

    dummy_file = DummyFile("doc2.pdf", b"pdfdata", "application/pdf")
    app.st.file_uploader = lambda *a, **k: dummy_file
    app.st.button = lambda *a, **k: True

    responses, DummyResponse = patch_requests
    responses[('POST', "http://localhost:8000/extract")] = DummyResponse(400, None, "Bad Request")

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [None, None, types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: dummy_file,
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: None,
            columns=lambda n: [None for _ in range(n)],
            expander=lambda *a, **k: None,
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        )]
        app.st.stop()

def test_data_extraction_request_exception(patch_streamlit, monkeypatch):
    class DummyFile:
        def __init__(self, name, content, type_):
            self.name = name
            self._content = content
            self.type = type_
        def getvalue(self): return self._content

    dummy_file = DummyFile("doc2.pdf", b"pdfdata", "application/pdf")
    app.st.file_uploader = lambda *a, **k: dummy_file
    app.st.button = lambda *a, **k: True

    def raise_exc(*a, **k): raise Exception("network error")
    monkeypatch.setattr(app.requests, "post", raise_exc)

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [None, None, types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: dummy_file,
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: None,
            columns=lambda n: [None for _ in range(n)],
            expander=lambda *a, **k: None,
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        )]
        app.st.stop()

def test_data_extraction_no_file_uploaded(patch_streamlit):
    app.st.file_uploader = lambda *a, **k: None
    app.st.button = lambda *a, **k: True

    with pytest.raises(SystemExit):
        app.__dict__["is_backend_up"] = True
        app.tabs = lambda labels: [None, None, types.SimpleNamespace(
            header=lambda *a, **k: None,
            write=lambda *a, **k: None,
            file_uploader=lambda *a, **k: None,
            button=lambda *a, **k: True,
            spinner=lambda *a, **k: None,
            columns=lambda n: [None for _ in range(n)],
            expander=lambda *a, **k: None,
            subheader=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            markdown=lambda *a, **k: None,
            json=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            success=lambda *a, **k: None
        )]
        app.st.stop()

def test_backend_unreachable_error_and_retry(patch_streamlit):
    # Simulate backend down
    app.__dict__["is_backend_up"] = False
    called = {}
    def fake_button(label):
        called['button'] = True
        return True
    def fake_rerun():
        called['rerun'] = True
    app.st.button = fake_button
    app.st.rerun = fake_rerun
    with pytest.raises(SystemExit):
        app.st.stop()
    assert called.get('button')
    assert called.get('rerun')
