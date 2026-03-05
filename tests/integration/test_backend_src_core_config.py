import os
import sys
import types
import tempfile
from pathlib import Path
import pytest

from backend.src.core.config import Settings

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Remove all relevant environment variables before each test for isolation
    for var in [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]:
        monkeypatch.delenv(var, raising=False)

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
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.75")
    s = Settings()
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

def test_settings_env_file_override(tmp_path, monkeypatch):
    env_content = (
        "APP_NAME=EnvFileApp\n"
        "API_V1_STR=/env/api\n"
        "GROQ_API_KEY=gsk_envfile_key\n"
        "QA_MODEL=envfile-qa-model\n"
        "VISION_MODEL=envfile-vision-model\n"
        "EMBEDDING_MODEL=envfile-embedding-model\n"
        "CHUNK_SIZE=4321\n"
        "CHUNK_OVERLAP=123\n"
        "TOP_K=9\n"
        "SIMILARITY_THRESHOLD=0.99\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    # Change working directory so .env is found
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.APP_NAME == "EnvFileApp"
    assert s.API_V1_STR == "/env/api"
    assert s.GROQ_API_KEY == "gsk_envfile_key"
    assert s.QA_MODEL == "envfile-qa-model"
    assert s.VISION_MODEL == "envfile-vision-model"
    assert s.EMBEDDING_MODEL == "envfile-embedding-model"
    assert s.CHUNK_SIZE == 4321
    assert s.CHUNK_OVERLAP == 123
    assert s.TOP_K == 9
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_env_file_and_env_var_precedence(tmp_path, monkeypatch):
    # .env file has one value, env var has another
    env_content = "APP_NAME=EnvFileApp\n"
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_NAME", "EnvVarApp")
    s = Settings()
    # Environment variable should take precedence over .env file
    assert s.APP_NAME == "EnvVarApp"

def test_settings_type_casting(monkeypatch):
    # Set int and float as strings
    monkeypatch.setenv("CHUNK_SIZE", "2048")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = Settings()
    assert s.CHUNK_SIZE == 2048
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_invalid_int(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        Settings()

def test_settings_invalid_float(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        Settings()

def test_settings_boundary_values(monkeypatch):
    # Test edge/boundary values for int and float fields
    monkeypatch.setenv("CHUNK_SIZE", str(2**31-1))
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "1")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = Settings()
    assert s.CHUNK_SIZE == 2**31-1
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 1
    assert s.SIMILARITY_THRESHOLD == 1.0

def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env vars due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase")
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_extra_env_ignored(monkeypatch):
    # Extra env vars not in Settings should be ignored due to extra="ignore"
    monkeypatch.setenv("UNRELATED_VAR", "should_be_ignored")
    s = Settings()
    assert not hasattr(s, "UNRELATED_VAR")

def test_settings_repr_and_str():
    s = Settings()
    # __repr__ and __str__ should include class name and fields
    r = repr(s)
    st = str(s)
    assert "Settings" in r
    assert "Settings" in st
    assert "APP_NAME" in r
    assert "APP_NAME" in st

def test_settings_equivalent_paths(monkeypatch, tmp_path):
    # Reconciliation: .env and direct env var with same value yield same result
    env_content = "QA_MODEL=llama-3.3-70b-versatile\n"
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    s1 = Settings()
    monkeypatch.setenv("QA_MODEL", "llama-3.3-70b-versatile")
    s2 = Settings()
    assert s1.QA_MODEL == s2.QA_MODEL
