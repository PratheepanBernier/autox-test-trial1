import os
import pytest
from unittest import mock
from backend.src.core import config

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear relevant environment variables before each test for isolation
    for var in [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]:
        monkeypatch.delenv(var, raising=False)

def test_settings_happy_path_defaults():
    # Test that default values are loaded when no env vars are set
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
    # Test that environment variables override defaults
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("API_V1_STR", "/test/api")
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
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.9

def test_settings_env_type_casting(monkeypatch):
    # Test that type casting from env vars works as expected
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "-1")
    monkeypatch.setenv("TOP_K", "999999")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = config.Settings()
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == -1
    assert s.TOP_K == 999999
    assert s.SIMILARITY_THRESHOLD == 1.0

def test_settings_env_invalid_type(monkeypatch):
    # Test that invalid type in env var raises a validation error
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        config.Settings()

def test_settings_env_boundary_conditions(monkeypatch):
    # Test boundary values for numeric fields
    monkeypatch.setenv("CHUNK_SIZE", str(2**31-1))
    monkeypatch.setenv("CHUNK_OVERLAP", str(-2**31))
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = config.Settings()
    assert s.CHUNK_SIZE == 2**31-1
    assert s.CHUNK_OVERLAP == -2**31
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_env_file_loading(monkeypatch, tmp_path):
    # Test that .env file is loaded if present
    env_content = (
        "APP_NAME=Env File App\n"
        "CHUNK_SIZE=321\n"
        "SIMILARITY_THRESHOLD=0.77\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    # Remove env vars to ensure .env is used
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    s = config.Settings()
    assert s.APP_NAME == "Env File App"
    assert s.CHUNK_SIZE == 321
    assert s.SIMILARITY_THRESHOLD == 0.77

def test_settings_case_sensitive(monkeypatch):
    # Test that environment variable names are case sensitive
    monkeypatch.setenv("app_name", "lowercase app")
    s = config.Settings()
    # Should not pick up lowercase env var
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_extra_fields_ignored(tmp_path, monkeypatch):
    # Test that extra fields in .env are ignored
    env_content = (
        "APP_NAME=Extra Field App\n"
        "EXTRA_FIELD=should_be_ignored\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    s = config.Settings()
    assert s.APP_NAME == "Extra Field App"
    # No attribute for EXTRA_FIELD
    assert not hasattr(s, "EXTRA_FIELD")

def test_settings_global_instance_consistency(monkeypatch):
    # Test that the global settings instance matches a new instance with same env
    monkeypatch.setenv("APP_NAME", "Consistent App")
    s1 = config.Settings()
    # Re-import to force global instance recreation
    import importlib
    import backend.src.core.config as config2
    s2 = config2.Settings()
    assert s1.dict() == s2.dict()

def test_settings_regression_default_vs_env(monkeypatch):
    # Regression: compare outputs for default and env override
    s_default = config.Settings()
    monkeypatch.setenv("APP_NAME", "Regression App")
    s_env = config.Settings()
    assert s_default.APP_NAME != s_env.APP_NAME
    assert s_env.APP_NAME == "Regression App"
    assert s_default.APP_NAME == "Logistics Document Intelligence Assistant"
