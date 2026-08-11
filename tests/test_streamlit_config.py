import pytest
from unittest.mock import Mock
from streamlit_app.config.streamlit_config import configure_page


def test_configure_page_sets_page_config(monkeypatch):
    import streamlit as st
    calls = {}
    monkeypatch.setattr(st, "set_page_config", lambda **kw: calls.update(kw))
    configure_page()
    assert calls["page_title"] == "APIS — Academic Performance Intelligence System"
    assert calls["page_icon"] == "🧠"
    assert calls["layout"] == "wide"
    assert calls["initial_sidebar_state"] == "expanded"