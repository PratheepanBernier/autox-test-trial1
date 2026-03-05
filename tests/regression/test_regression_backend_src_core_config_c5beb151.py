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
    monkeypatch.setenv("CHUNK_SIZE", str(2**31-1))
    s = config.Settings()
    assert s.CHUNK_SIZE == 2**31-1


def test_settings_boundary_similarity_threshold(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = config.Settings()
    assert s.SIMILARITY_THRESHOLD == 0.0
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = config.Settings()
    assert s.SIMILARITY_THRESHOLD == 1.0


def test_settings_extra_env_ignored(monkeypatch):
    monkeypatch.setenv("NON_EXISTENT_SETTING", "should_be_ignored")
    s = config.Settings()
    assert not hasattr(s, "NON_EXISTENT_SETTING")


def test_settings_case_sensitive(monkeypatch):
    monkeypatch.setenv("app_name", "lowercase_should_not_override")
    s = config.Settings()
    # Should not override because of case sensitivity
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"


def test_settings_env_file_loading(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=FromEnvFile\nCHUNK_SIZE=4321\n")
    with mock.patch.object(config.Settings.Config, "env_file", str(env_file)):
        s = config.Settings(_env_file=str(env_file))
        assert s.APP_NAME == "FromEnvFile"
        assert s.CHUNK_SIZE == 4321


def test_settings_repr_and_str():
    s = config.Settings()
    r = repr(s)
    st = str(s)
    assert "Settings" in r
    assert "Settings" in st


def test_settings_instance_is_singleton_like():
    s1 = config.settings
    s2 = config.settings
    assert s1 is s2
    assert s1.APP_NAME == s2.APP_NAME
