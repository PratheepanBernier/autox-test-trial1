import os
import sys
import types
import tempfile
import shutil
from unittest import mock
import pytest

# Ensure import path is correct for backend/src/core/config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core import config

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear relevant environment variables before each test for isolation
    keys = [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

def test_settings_default_values_are_correct():
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
    monkeypatch.setenv("APP_NAME", "TestApp")
    monkeypatch.setenv("API_V1_STR", "/test/api")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("CHUNK_OVERLAP", "45")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    s = config.Settings()
    assert s.APP_NAME == "TestApp"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_env_file_override(tmp_path, monkeypatch):
    env_content = (
        "APP_NAME=EnvFileApp\n"
        "API_V1_STR=/env/api\n"
        "GROQ_API_KEY=gsk_env\n"
        "QA_MODEL=env-qa-model\n"
        "VISION_MODEL=env-vision-model\n"
        "EMBEDDING_MODEL=env-embedding-model\n"
        "CHUNK_SIZE=321\n"
        "CHUNK_OVERLAP=54\n"
        "TOP_K=8\n"
        "SIMILARITY_THRESHOLD=0.88\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    # Re-import config.Settings to pick up new env file location
    import importlib
    import core.config as config_reload
    importlib.reload(config_reload)
    s = config_reload.Settings()
    assert s.APP_NAME == "EnvFileApp"
    assert s.API_V1_STR == "/env/api"
    assert s.GROQ_API_KEY == "gsk_env"
    assert s.QA_MODEL == "env-qa-model"
    assert s.VISION_MODEL == "env-vision-model"
    assert s.EMBEDDING_MODEL == "env-embedding-model"
    assert s.CHUNK_SIZE == 321
    assert s.CHUNK_OVERLAP == 54
    assert s.TOP_K == 8
    assert s.SIMILARITY_THRESHOLD == 0.88

def test_settings_env_var_precedence_over_env_file(tmp_path, monkeypatch):
    env_content = (
        "APP_NAME=EnvFileApp\n"
        "CHUNK_SIZE=111\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_NAME", "EnvVarApp")
    monkeypatch.setenv("CHUNK_SIZE", "222")
    import importlib
    import core.config as config_reload
    importlib.reload(config_reload)
    s = config_reload.Settings()
    assert s.APP_NAME == "EnvVarApp"
    assert s.CHUNK_SIZE == 222

def test_settings_type_coercion_and_error(monkeypatch):
    # Valid coercion
    monkeypatch.setenv("CHUNK_SIZE", "42")
    s = config.Settings()
    assert s.CHUNK_SIZE == 42
    # Invalid coercion should raise ValueError
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        config.Settings()

def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env var due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase")
    s = config.Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_extra_fields_ignored(monkeypatch):
    monkeypatch.setenv("EXTRA_FIELD", "should_be_ignored")
    s = config.Settings()
    assert not hasattr(s, "EXTRA_FIELD")

def test_settings_equivalent_paths_consistency(monkeypatch):
    # Reconciliation: instantiating via config.settings and config.Settings() should yield same values
    monkeypatch.setenv("QA_MODEL", "recon-qa-model")
    s1 = config.Settings()
    # config.settings is instantiated at import time, so we need to reload to pick up env change
    import importlib
    import core.config as config_reload
    importlib.reload(config_reload)
    s2 = config_reload.settings
    assert s1.QA_MODEL == s2.QA_MODEL == "recon-qa-model"

def test_settings_boundary_conditions(monkeypatch):
    # Test boundary values for numeric fields
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = config.Settings()
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

    monkeypatch.setenv("CHUNK_SIZE", str(2**31-1))
    monkeypatch.setenv("CHUNK_OVERLAP", str(2**31-1))
    monkeypatch.setenv("TOP_K", str(2**31-1))
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = config.Settings()
    assert s.CHUNK_SIZE == 2**31-1
    assert s.CHUNK_OVERLAP == 2**31-1
    assert s.TOP_K == 2**31-1
    assert s.SIMILARITY_THRESHOLD == 1.0
