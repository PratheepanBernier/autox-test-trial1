import pytest
import requests
import time
import subprocess
import os

def test_ping():
    """Simple ping test to verify backend is reachable if running."""
    # This test is a placeholder for the actual end-to-end tests
    url = os.getenv("BACKEND_URL", "http://localhost:8000")
    try:
        response = requests.get(f"{url}/ping", timeout=5)
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend is not running")

def test_streamlit_app_exists():
    assert os.path.exists("streamlit_app.py")
    assert os.path.exists("frontend/app.py")
    assert os.path.exists("backend/main.py")
