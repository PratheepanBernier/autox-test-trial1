import os
import pytest
from unittest import mock
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

def test_settings_env_override(monkeypatch):
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
    s = Settings()
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

def test_settings_partial_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Partial App")
    s = Settings()
    assert s.APP_NAME == "Partial App"
    assert s.API_V1_STR == "/api/v1"  # default
    assert s.GROQ_API_KEY == "gsk_your_api_key"  # default

def test_settings_invalid_types(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        Settings()

def test_settings_config_class_attributes():
    # Accessing Config class attributes
    assert Settings.Config.case_sensitive is True
    assert Settings.Config.env_file == ".env"
    assert Settings.Config.extra == "ignore"

def test_settings_env_file_loading(tmp_path, monkeypatch):
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=EnvFileApp\nCHUNK_SIZE=321\n")
    monkeypatch.chdir(tmp_path)
    # Patch Settings.Config.env_file to point to our temp .env
    with mock.patch.object(Settings.Config, "env_file", str(env_file)):
        s = Settings()
        assert s.APP_NAME == "EnvFileApp"
        assert s.CHUNK_SIZE == 321

def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env var if case_sensitive is True
    monkeypatch.setenv("app_name", "lowercase")
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_extra_fields_ignored(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=ExtraFieldApp\nEXTRA_FIELD=should_be_ignored\n")
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(Settings.Config, "env_file", str(env_file)):
        s = Settings()
        assert s.APP_NAME == "ExtraFieldApp"
        # Should not have attribute for extra field
        assert not hasattr(s, "EXTRA_FIELD")
