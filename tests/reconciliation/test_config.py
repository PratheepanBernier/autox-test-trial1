import pytest
from unittest import mock
from pydantic_settings import BaseSettings
from backend.src.core import config

def test_settings_happy_path_equivalent_instantiation():
    # Reconcile: direct instantiation vs. module-level instance
    s1 = config.Settings()
    s2 = config.settings
    assert s1.dict() == s2.dict()

def test_settings_env_override(monkeypatch):
    # Reconcile: env var override vs. default
    monkeypatch.setenv("APP_NAME", "Test App")
    s_env = config.Settings()
    assert s_env.APP_NAME == "Test App"
    # Should not affect module-level instance (already instantiated)
    assert config.settings.APP_NAME != "Test App"

def test_settings_env_override_equivalence(monkeypatch):
    # Reconcile: two instances with same env var
    monkeypatch.setenv("QA_MODEL", "test-model")
    s1 = config.Settings()
    s2 = config.Settings()
    assert s1.QA_MODEL == s2.QA_MODEL == "test-model"

def test_settings_default_values():
    # Reconcile: default values match expected
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

def test_settings_type_enforcement():
    # Reconcile: type enforcement for int/float
    s = config.Settings(CHUNK_SIZE=1234, CHUNK_OVERLAP=0, TOP_K=1, SIMILARITY_THRESHOLD=0.0)
    assert isinstance(s.CHUNK_SIZE, int)
    assert isinstance(s.CHUNK_OVERLAP, int)
    assert isinstance(s.TOP_K, int)
    assert isinstance(s.SIMILARITY_THRESHOLD, float)
    assert s.CHUNK_SIZE == 1234
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 1
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_invalid_type_raises():
    # Reconcile: error handling for invalid types
    with pytest.raises(ValueError):
        config.Settings(CHUNK_SIZE="not_an_int")
    with pytest.raises(ValueError):
        config.Settings(SIMILARITY_THRESHOLD="not_a_float")

def test_settings_partial_override():
    # Reconcile: partial override keeps other defaults
    s = config.Settings(APP_NAME="Custom Name")
    assert s.APP_NAME == "Custom Name"
    assert s.API_V1_STR == "/api/v1"  # default

def test_settings_config_class_attributes():
    # Reconcile: Config class attributes
    assert config.Settings.Config.case_sensitive is True
    assert config.Settings.Config.env_file == ".env"
    assert config.Settings.Config.extra == "ignore"

def test_settings_repr_and_str_equivalence():
    # Reconcile: __repr__ and __str__ are consistent for two identical instances
    s1 = config.Settings()
    s2 = config.Settings()
    assert repr(s1) == repr(s2)
    assert str(s1) == str(s2)

def test_settings_edge_case_empty_strings():
    # Reconcile: empty string override
    s = config.Settings(APP_NAME="", QA_MODEL="")
    assert s.APP_NAME == ""
    assert s.QA_MODEL == ""

def test_settings_boundary_chunk_size(monkeypatch):
    # Reconcile: boundary values for CHUNK_SIZE
    s_min = config.Settings(CHUNK_SIZE=0)
    s_max = config.Settings(CHUNK_SIZE=2**31-1)
    assert s_min.CHUNK_SIZE == 0
    assert s_max.CHUNK_SIZE == 2**31-1

def test_settings_boundary_similarity_threshold():
    # Reconcile: boundary values for SIMILARITY_THRESHOLD
    s_zero = config.Settings(SIMILARITY_THRESHOLD=0.0)
    s_one = config.Settings(SIMILARITY_THRESHOLD=1.0)
    assert s_zero.SIMILARITY_THRESHOLD == 0.0
    assert s_one.SIMILARITY_THRESHOLD == 1.0

def test_settings_extra_fields_ignored():
    # Reconcile: extra fields are ignored due to extra="ignore"
    s = config.Settings(APP_NAME="A", extra_field="should_be_ignored")
    assert not hasattr(s, "extra_field")
    assert s.APP_NAME == "A"

def test_settings_case_sensitive_env(monkeypatch):
    # Reconcile: case sensitivity in env vars
    monkeypatch.setenv("app_name", "lowercase")
    s = config.Settings()
    # Should not pick up lowercase env var due to case_sensitive=True
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_equivalence_with_dict(monkeypatch):
    # Reconcile: instantiation from dict vs. kwargs
    values = {
        "APP_NAME": "DictName",
        "API_V1_STR": "/dict/api",
        "GROQ_API_KEY": "dict_key",
        "QA_MODEL": "dict_model",
        "VISION_MODEL": "dict_vision",
        "EMBEDDING_MODEL": "dict_embed",
        "CHUNK_SIZE": 111,
        "CHUNK_OVERLAP": 22,
        "TOP_K": 3,
        "SIMILARITY_THRESHOLD": 0.9,
    }
    s1 = config.Settings(**values)
    s2 = config.Settings.parse_obj(values)
    assert s1.dict() == s2.dict()
