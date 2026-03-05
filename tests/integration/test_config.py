# source_hash: 19f86d3713912848
import os
import pytest
from unittest import mock
from backend.src.core.config import Settings

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear relevant environment variables before each test for isolation
    env_vars = [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield

def test_settings_default_values():
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.CHUNK_SIZE == 1000
    assert s.CHUNK_OVERLAP == 200
    assert s.TOP_K == 4
    assert s.SIMILARITY_THRESHOLD == 0.5

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("API_V1_STR", "/test/api")
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("CHUNK_OVERLAP", "45")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    s = Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_partial_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Partial App")
    s = Settings()
    assert s.APP_NAME == "Partial App"
    # All others should be defaults
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"

def test_settings_invalid_int_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        Settings()

def test_settings_invalid_float_env(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        Settings()

def test_settings_boundary_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "0")
    s = Settings()
    assert s.CHUNK_SIZE == 0
    monkeypatch.setenv("CHUNK_SIZE", "1")
    s = Settings()
    assert s.CHUNK_SIZE == 1
    monkeypatch.setenv("CHUNK_SIZE", str(2**31-1))
    s = Settings()
    assert s.CHUNK_SIZE == 2**31-1

def test_settings_boundary_similarity_threshold(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = Settings()
    assert s.SIMILARITY_THRESHOLD == 0.0
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = Settings()
    assert s.SIMILARITY_THRESHOLD == 1.0

def test_settings_extra_env_ignored(monkeypatch):
    monkeypatch.setenv("EXTRA_VAR", "should_be_ignored")
    s = Settings()
    assert not hasattr(s, "EXTRA_VAR")

def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env var due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase")
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_env_file_loading(monkeypatch, tmp_path):
    # Simulate .env file loading
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=EnvFileApp\nCHUNK_SIZE=321\n")
    with mock.patch("pydantic_settings.env_settings_source") as env_settings_source_mock:
        # Patch to simulate .env file loading
        env_settings_source_mock.return_value = {"APP_NAME": "EnvFileApp", "CHUNK_SIZE": 321}
        s = Settings(_env_file=str(env_file))
        assert s.APP_NAME == "EnvFileApp"
        assert s.CHUNK_SIZE == 321

def test_settings_equivalent_paths(monkeypatch):
    # Reconciliation: env var and direct instantiation should match
    monkeypatch.setenv("APP_NAME", "ReconApp")
    s_env = Settings()
    s_direct = Settings(APP_NAME="ReconApp")
    assert s_env.APP_NAME == s_direct.APP_NAME
    # Other fields should be default
    assert s_env.QA_MODEL == s_direct.QA_MODEL

def test_settings_error_on_missing_required(monkeypatch):
    # All fields have defaults, so no error should be raised if nothing is set
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"
