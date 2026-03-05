import os
import pytest
from unittest import mock
from backend.src.core import config

def test_settings_default_values():
    # Test that default values are set correctly
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
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.75")
    s = config.Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 1234
    assert s.CHUNK_OVERLAP == 321
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.75

def test_settings_env_file(monkeypatch, tmp_path):
    # Test that .env file is loaded and overrides defaults
    env_content = (
        "APP_NAME=Env File App\n"
        "API_V1_STR=/env/api\n"
        "GROQ_API_KEY=gsk_env_key\n"
        "QA_MODEL=env-qa-model\n"
        "VISION_MODEL=env-vision-model\n"
        "EMBEDDING_MODEL=env-embedding-model\n"
        "CHUNK_SIZE=4321\n"
        "CHUNK_OVERLAP=123\n"
        "TOP_K=9\n"
        "SIMILARITY_THRESHOLD=0.99\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    # Clear relevant env vars to ensure .env is used
    for var in [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL",
        "VISION_MODEL", "EMBEDDING_MODEL", "CHUNK_SIZE",
        "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]:
        monkeypatch.delenv(var, raising=False)
    s = config.Settings()
    assert s.APP_NAME == "Env File App"
    assert s.API_V1_STR == "/env/api"
    assert s.GROQ_API_KEY == "gsk_env_key"
    assert s.QA_MODEL == "env-qa-model"
    assert s.VISION_MODEL == "env-vision-model"
    assert s.EMBEDDING_MODEL == "env-embedding-model"
    assert s.CHUNK_SIZE == 4321
    assert s.CHUNK_OVERLAP == 123
    assert s.TOP_K == 9
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_type_casting(monkeypatch):
    # Test that type casting works for int and float fields
    monkeypatch.setenv("CHUNK_SIZE", "999")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = config.Settings()
    assert s.CHUNK_SIZE == 999
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_invalid_type(monkeypatch):
    # Test that invalid type raises a validation error
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(Exception):
        config.Settings()

def test_settings_boundary_conditions(monkeypatch):
    # Test boundary values for numeric fields
    monkeypatch.setenv("CHUNK_SIZE", "1")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "1")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = config.Settings()
    assert s.CHUNK_SIZE == 1
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 1
    assert s.SIMILARITY_THRESHOLD == 1.0

def test_settings_config_class_attributes():
    # Test that Config class attributes are set as expected
    assert config.Settings.Config.case_sensitive is True
    assert config.Settings.Config.env_file == ".env"
    assert config.Settings.Config.extra == "ignore"

def test_settings_equivalent_paths(monkeypatch, tmp_path):
    # Test reconciliation: env var and .env file with same value yield same result
    monkeypatch.setenv("APP_NAME", "Recon App")
    s_env = config.Settings()
    monkeypatch.delenv("APP_NAME", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=Recon App\n")
    monkeypatch.chdir(tmp_path)
    s_file = config.Settings()
    assert s_env.APP_NAME == s_file.APP_NAME

def test_settings_preserves_regression_behavior():
    # Test that the global settings object has expected default values (regression)
    s = config.settings
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
