# source_hash: 19f86d3713912848
import pytest
from unittest import mock
from pydantic_settings import BaseSettings
from backend.src.core import config

def test_settings_happy_path_and_defaults():
    # Ensure default values are set correctly
    s = config.Settings()
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
    # Override some environment variables and check they are picked up
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("API_V1_STR", "/test/v1")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("CHUNK_OVERLAP", "45")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.9")
    s = config.Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/v1"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.9

def test_settings_type_enforcement():
    # Type errors should be raised for invalid types
    with pytest.raises(ValueError):
        config.Settings(CHUNK_SIZE="not_an_int")
    with pytest.raises(ValueError):
        config.Settings(SIMILARITY_THRESHOLD="not_a_float")
    with pytest.raises(ValueError):
        config.Settings(TOP_K="not_an_int")

def test_settings_boundary_conditions():
    # Test boundary values for numeric fields
    s = config.Settings(CHUNK_SIZE=0, CHUNK_OVERLAP=0, TOP_K=0, SIMILARITY_THRESHOLD=0.0)
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

    s2 = config.Settings(CHUNK_SIZE=2**31-1, CHUNK_OVERLAP=2**31-1, TOP_K=2**31-1, SIMILARITY_THRESHOLD=1.0)
    assert s2.CHUNK_SIZE == 2**31-1
    assert s2.CHUNK_OVERLAP == 2**31-1
    assert s2.TOP_K == 2**31-1
    assert s2.SIMILARITY_THRESHOLD == 1.0

def test_settings_extra_fields_are_ignored():
    # Extra fields should be ignored due to extra="ignore"
    s = config.Settings(**{"APP_NAME": "X", "EXTRA_FIELD": "should_be_ignored"})
    assert not hasattr(s, "EXTRA_FIELD")
    assert s.APP_NAME == "X"

def test_settings_case_sensitive(monkeypatch):
    # Case sensitivity: lower-case env var should not override
    monkeypatch.setenv("app_name", "lowercase")
    s = config.Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_env_file_loading(monkeypatch, tmp_path):
    # Simulate .env file loading by patching BaseSettings._build_values
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=FromEnvFile\n")
    with mock.patch.object(config.Settings.Config, "env_file", str(env_file)):
        s = config.Settings(_env_file=str(env_file))
        assert s.APP_NAME == "FromEnvFile"

def test_settings_equivalent_paths(monkeypatch):
    # Reconciliation: instantiating via config.settings and direct Settings() should yield same values
    monkeypatch.delenv("APP_NAME", raising=False)
    s1 = config.Settings()
    s2 = config.settings
    assert s1.dict() == s2.dict()

def test_settings_regression_preserves_behavior():
    # Regression: ensure that changing a default value is detected
    s = config.Settings()
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
