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

from backend.src.core import config

def test_settings_default_values():
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
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("API_V1_STR", "/test/api")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("CHUNK_OVERLAP", "45")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    s = config.Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 9
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_partial_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Partial App")
    s = config.Settings()
    assert s.APP_NAME == "Partial App"
    assert s.API_V1_STR == "/api/v1"  # default
    assert s.GROQ_API_KEY == "gsk_your_api_key"  # default

def test_settings_invalid_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        config.Settings()

def test_settings_invalid_similarity_threshold(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        config.Settings()

def test_settings_boundary_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "0")
    s = config.Settings()
    assert s.CHUNK_SIZE == 0

    monkeypatch.setenv("CHUNK_SIZE", "-1")
    s2 = config.Settings()
    assert s2.CHUNK_SIZE == -1

def test_settings_boundary_similarity_threshold(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = config.Settings()
    assert s.SIMILARITY_THRESHOLD == 0.0

    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s2 = config.Settings()
    assert s2.SIMILARITY_THRESHOLD == 1.0

def test_settings_config_class_attributes():
    assert config.Settings.Config.case_sensitive is True
    assert config.Settings.Config.env_file == ".env"
    assert config.Settings.Config.extra == "ignore"

def test_settings_repr_and_str():
    s = config.Settings()
    r = repr(s)
    st = str(s)
    assert "Settings" in r
    assert "Settings" in st

def test_settings_equivalent_paths(monkeypatch):
    # Reconciliation: env and direct instantiation should match if same values
    monkeypatch.setenv("APP_NAME", "Recon App")
    monkeypatch.setenv("CHUNK_SIZE", "555")
    s_env = config.Settings()
    s_direct = config.Settings(APP_NAME="Recon App", CHUNK_SIZE=555)
    assert s_env.APP_NAME == s_direct.APP_NAME
    assert s_env.CHUNK_SIZE == s_direct.CHUNK_SIZE

def test_settings_error_on_extra_fields():
    # Extra fields should be ignored, not raise error
    s = config.Settings(unknown_field="should be ignored")
    assert not hasattr(s, "unknown_field")

def test_settings_instance_isolation():
    s1 = config.Settings(APP_NAME="A")
    s2 = config.Settings(APP_NAME="B")
    assert s1.APP_NAME == "A"
    assert s2.APP_NAME == "B"
    assert s1 is not s2

def test_global_settings_instance_type():
    assert isinstance(config.settings, config.Settings)
    assert config.settings.APP_NAME == "Logistics Document Intelligence Assistant"
