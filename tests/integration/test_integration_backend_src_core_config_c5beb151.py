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

import os
import pytest
from unittest import mock

from backend.src.core import config


def test_settings_default_values_are_set():
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
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")

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
    assert s.SIMILARITY_THRESHOLD == 0.99


def test_settings_env_file_loading(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join([
            "APP_NAME=EnvFileApp",
            "API_V1_STR=/env/api",
            "GROQ_API_KEY=gsk_envfile_key",
            "QA_MODEL=envfile-qa-model",
            "VISION_MODEL=envfile-vision-model",
            "EMBEDDING_MODEL=envfile-embedding-model",
            "CHUNK_SIZE=4321",
            "CHUNK_OVERLAP=123",
            "TOP_K=9",
            "SIMILARITY_THRESHOLD=0.77"
        ])
    )
    monkeypatch.chdir(tmp_path)
    # Remove env vars to ensure only .env is used
    for var in [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]:
        monkeypatch.delenv(var, raising=False)

    s = config.Settings()
    assert s.APP_NAME == "EnvFileApp"
    assert s.API_V1_STR == "/env/api"
    assert s.GROQ_API_KEY == "gsk_envfile_key"
    assert s.QA_MODEL == "envfile-qa-model"
    assert s.VISION_MODEL == "envfile-vision-model"
    assert s.EMBEDDING_MODEL == "envfile-embedding-model"
    assert s.CHUNK_SIZE == 4321
    assert s.CHUNK_OVERLAP == 123
    assert s.TOP_K == 9
    assert s.SIMILARITY_THRESHOLD == 0.77


def test_settings_type_casting_and_invalid_values(monkeypatch):
    # Valid integer and float values as strings
    monkeypatch.setenv("CHUNK_SIZE", "555")
    monkeypatch.setenv("CHUNK_OVERLAP", "111")
    monkeypatch.setenv("TOP_K", "2")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.33")
    s = config.Settings()
    assert s.CHUNK_SIZE == 555
    assert s.CHUNK_OVERLAP == 111
    assert s.TOP_K == 2
    assert s.SIMILARITY_THRESHOLD == 0.33

    # Invalid integer value
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        config.Settings()

    # Invalid float value
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        config.Settings()


def test_settings_config_class_attributes():
    assert config.Settings.Config.case_sensitive is True
    assert config.Settings.Config.env_file == ".env"
    assert config.Settings.Config.extra == "ignore"


def test_settings_extra_fields_are_ignored(monkeypatch):
    monkeypatch.setenv("EXTRA_FIELD", "should_be_ignored")
    s = config.Settings()
    assert not hasattr(s, "EXTRA_FIELD")


def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env var due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase_app")
    s = config.Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"


def test_settings_instance_is_singleton_like():
    s1 = config.settings
    s2 = config.settings
    assert s1 is s2
    assert s1.APP_NAME == s2.APP_NAME
    assert s1.QA_MODEL == s2.QA_MODEL
