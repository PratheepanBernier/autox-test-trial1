# source_hash: 19f86d3713912848
import pytest
from unittest import mock
from pydantic_settings import BaseSettings
from backend.src.core.config import Settings

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

def test_settings_override_with_kwargs():
    s = Settings(
        APP_NAME="Test App",
        API_V1_STR="/test/api",
        GROQ_API_KEY="override_key",
        QA_MODEL="test-model",
        VISION_MODEL="test-vision",
        EMBEDDING_MODEL="test-embedding",
        CHUNK_SIZE=123,
        CHUNK_OVERLAP=45,
        TOP_K=2,
        SIMILARITY_THRESHOLD=0.9
    )
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "override_key"
    assert s.QA_MODEL == "test-model"
    assert s.VISION_MODEL == "test-vision"
    assert s.EMBEDDING_MODEL == "test-embedding"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 2
    assert s.SIMILARITY_THRESHOLD == 0.9

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Env App")
    monkeypatch.setenv("API_V1_STR", "/env/api")
    monkeypatch.setenv("GROQ_API_KEY", "env_key")
    monkeypatch.setenv("QA_MODEL", "env-model")
    monkeypatch.setenv("VISION_MODEL", "env-vision")
    monkeypatch.setenv("EMBEDDING_MODEL", "env-embedding")
    monkeypatch.setenv("CHUNK_SIZE", "321")
    monkeypatch.setenv("CHUNK_OVERLAP", "54")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.8")
    s = Settings()
    assert s.APP_NAME == "Env App"
    assert s.API_V1_STR == "/env/api"
    assert s.GROQ_API_KEY == "env_key"
    assert s.QA_MODEL == "env-model"
    assert s.VISION_MODEL == "env-vision"
    assert s.EMBEDDING_MODEL == "env-embedding"
    assert s.CHUNK_SIZE == 321
    assert s.CHUNK_OVERLAP == 54
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.8

def test_settings_invalid_types_raise_validation_error():
    with pytest.raises(ValueError):
        Settings(CHUNK_SIZE="not_an_int")
    with pytest.raises(ValueError):
        Settings(CHUNK_OVERLAP="not_an_int")
    with pytest.raises(ValueError):
        Settings(TOP_K="not_an_int")
    with pytest.raises(ValueError):
        Settings(SIMILARITY_THRESHOLD="not_a_float")

def test_settings_boundary_conditions():
    s = Settings(CHUNK_SIZE=0, CHUNK_OVERLAP=0, TOP_K=0, SIMILARITY_THRESHOLD=0.0)
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

    s = Settings(CHUNK_SIZE=2**31-1, CHUNK_OVERLAP=2**31-1, TOP_K=2**31-1, SIMILARITY_THRESHOLD=1.0)
    assert s.CHUNK_SIZE == 2**31-1
    assert s.CHUNK_OVERLAP == 2**31-1
    assert s.TOP_K == 2**31-1
    assert s.SIMILARITY_THRESHOLD == 1.0

def test_settings_config_class_attributes():
    assert Settings.Config.case_sensitive is True
    assert Settings.Config.env_file == ".env"
    assert Settings.Config.extra == "ignore"

def test_settings_extra_fields_are_ignored():
    s = Settings(**{"APP_NAME": "Extra Test", "EXTRA_FIELD": "should be ignored"})
    assert s.APP_NAME == "Extra Test"
    assert not hasattr(s, "EXTRA_FIELD")

def test_settings_case_sensitive_env(monkeypatch):
    # Should not pick up lower-case env var if case_sensitive is True
    monkeypatch.setenv("app_name", "lowercase")
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_repr_and_str_are_consistent():
    s = Settings()
    assert str(s) == repr(s)
    assert "APP_NAME" in str(s)
    assert "GROQ_API_KEY" in str(s)
    assert "CHUNK_SIZE" in str(s)
    assert "SIMILARITY_THRESHOLD" in str(s)
