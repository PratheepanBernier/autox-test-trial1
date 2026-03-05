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
from backend.src.core import config as config_module

@pytest.fixture
def default_settings():
    # Return a fresh Settings instance with default values
    return config_module.Settings()

def test_settings_happy_path_and_defaults(default_settings):
    s = default_settings
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
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    s = config_module.Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_type_enforcement():
    # Should raise error if wrong type is provided
    with pytest.raises(ValueError):
        config_module.Settings(CHUNK_SIZE="not_an_int")
    with pytest.raises(ValueError):
        config_module.Settings(SIMILARITY_THRESHOLD="not_a_float")
    with pytest.raises(ValueError):
        config_module.Settings(TOP_K="not_an_int")

def test_settings_boundary_conditions():
    # Test boundary values for chunk size, overlap, top_k, similarity threshold
    s = config_module.Settings(CHUNK_SIZE=0, CHUNK_OVERLAP=0, TOP_K=0, SIMILARITY_THRESHOLD=0.0)
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

    s2 = config_module.Settings(CHUNK_SIZE=2**31-1, CHUNK_OVERLAP=2**31-1, TOP_K=2**31-1, SIMILARITY_THRESHOLD=1.0)
    assert s2.CHUNK_SIZE == 2**31-1
    assert s2.CHUNK_OVERLAP == 2**31-1
    assert s2.TOP_K == 2**31-1
    assert s2.SIMILARITY_THRESHOLD == 1.0

def test_settings_config_class_attributes():
    # Check Config inner class attributes
    assert config_module.Settings.Config.case_sensitive is True
    assert config_module.Settings.Config.env_file == ".env"
    assert config_module.Settings.Config.extra == "ignore"

def test_settings_repr_and_eq():
    s1 = config_module.Settings()
    s2 = config_module.Settings()
    assert repr(s1) == repr(s2)
    assert s1 == s2

def test_settings_env_file_loading(monkeypatch, tmp_path):
    # Simulate .env file loading by patching BaseSettings._build_values
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=EnvFileApp\nCHUNK_SIZE=555\n")
    with mock.patch.object(config_module.Settings.Config, "env_file", str(env_file)):
        s = config_module.Settings(_env_file=str(env_file))
        assert s.APP_NAME == "EnvFileApp"
        assert s.CHUNK_SIZE == 555

def test_settings_extra_fields_ignored():
    # Extra fields should be ignored due to extra="ignore"
    s = config_module.Settings(**{"APP_NAME": "X", "EXTRA_FIELD": "should_be_ignored"})
    assert not hasattr(s, "EXTRA_FIELD")
    assert s.APP_NAME == "X"

def test_settings_equivalent_paths_consistency(monkeypatch):
    # Reconciliation: settings via env vs direct instantiation
    monkeypatch.setenv("APP_NAME", "ReconApp")
    s_env = config_module.Settings()
    s_direct = config_module.Settings(APP_NAME="ReconApp")
    assert s_env.APP_NAME == s_direct.APP_NAME

def test_settings_invalid_env(monkeypatch):
    # If env var is invalid type, should raise error
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        config_module.Settings()

def test_settings_partial_env(monkeypatch):
    # Only some env vars set, rest should use defaults
    monkeypatch.setenv("APP_NAME", "PartialEnvApp")
    s = config_module.Settings()
    assert s.APP_NAME == "PartialEnvApp"
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.CHUNK_SIZE == 1000
    assert s.CHUNK_OVERLAP == 200
    assert s.TOP_K == 4
    assert s.SIMILARITY_THRESHOLD == 0.5

def test_settings_global_instance_consistency():
    # The global settings instance should match a fresh instance with no env
    s_global = config_module.settings
    s_fresh = config_module.Settings()
    assert s_global == s_fresh
