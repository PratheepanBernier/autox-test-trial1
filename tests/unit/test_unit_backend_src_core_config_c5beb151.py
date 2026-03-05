# source_hash: 19f86d3713912848
# import_target: backend.src.core.config
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest import mock

from backend.src.core.config import Settings, settings

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
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
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
    assert s.SIMILARITY_THRESHOLD == 0.99

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
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    # Remove CHUNK_SIZE to test SIMILARITY_THRESHOLD error
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    with pytest.raises(ValueError):
        Settings()

def test_settings_boundary_values(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = Settings()
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

    monkeypatch.setenv("CHUNK_SIZE", str(2**31-1))
    monkeypatch.setenv("CHUNK_OVERLAP", str(2**31-1))
    monkeypatch.setenv("TOP_K", str(2**31-1))
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = Settings()
    assert s.CHUNK_SIZE == 2**31-1
    assert s.CHUNK_OVERLAP == 2**31-1
    assert s.TOP_K == 2**31-1
    assert s.SIMILARITY_THRESHOLD == 1.0

def test_settings_config_class_attributes():
    assert hasattr(Settings.Config, "case_sensitive")
    assert Settings.Config.case_sensitive is True
    assert hasattr(Settings.Config, "env_file")
    assert Settings.Config.env_file == ".env"
    assert hasattr(Settings.Config, "extra")
    assert Settings.Config.extra == "ignore"

def test_settings_repr_and_str():
    s = Settings()
    r = repr(s)
    st = str(s)
    assert "Settings" in r
    assert "Settings" in st

def test_settings_isolation_between_instances():
    s1 = Settings(APP_NAME="A")
    s2 = Settings(APP_NAME="B")
    assert s1.APP_NAME == "A"
    assert s2.APP_NAME == "B"

def test_settings_global_instance_is_settings_class():
    assert isinstance(settings, Settings)
    assert settings.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_env_file_loading(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=FromEnvFile\nCHUNK_SIZE=4321\n")
    with mock.patch("backend.src.core.config.Settings.Config.env_file", str(env_file)):
        # Pydantic reads .env file only if present in current working directory or as specified in env_file
        # So we change cwd to tmp_path for this test
        with mock.patch("os.getcwd", return_value=str(tmp_path)):
            s = Settings(_env_file=str(env_file))
            assert s.APP_NAME == "FromEnvFile"
            assert s.CHUNK_SIZE == 4321

def test_settings_extra_fields_ignored():
    s = Settings(**{"APP_NAME": "X", "EXTRA_FIELD": "should be ignored"})
    assert s.APP_NAME == "X"
    assert not hasattr(s, "EXTRA_FIELD")
