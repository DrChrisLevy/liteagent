"""Shared pytest fixtures for litellm mocking."""

import pytest

from tests.mock_helpers import async_iter


@pytest.fixture
def mock_llm(monkeypatch):
    """Single-turn litellm mock. Returns captured kwargs dict."""
    captured = {}

    def _mock(chunks, final):
        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return async_iter(chunks)

        monkeypatch.setattr("liteagent.loop.litellm.acompletion", fake_acompletion)
        monkeypatch.setattr(
            "liteagent.loop.litellm.stream_chunk_builder", lambda _: final
        )
        return captured

    return _mock


@pytest.fixture
def mock_llm_seq(monkeypatch):
    """Multi-turn litellm mock. Takes list of (chunks, final) per LLM call."""

    def _mock(turns):
        remaining_chunks = [t[0] for t in turns]
        remaining_finals = [t[1] for t in turns]

        async def fake_acompletion(**kwargs):
            return async_iter(remaining_chunks.pop(0))

        monkeypatch.setattr("liteagent.loop.litellm.acompletion", fake_acompletion)
        monkeypatch.setattr(
            "liteagent.loop.litellm.stream_chunk_builder",
            lambda _: remaining_finals.pop(0),
        )

    return _mock
