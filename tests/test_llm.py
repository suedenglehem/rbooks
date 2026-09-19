"""M5 answer-model adapter (PRD §12): the protocol fake, the bounded-timeout
llama.cpp client (contracted payload, unavailable-vs-malformed error split),
and the never-raises factory the CLI and the application both use.

The model is optional by design: an unconfigured revision yields ``None``
(search keeps working), a server that is down is *unavailable* (transient,
surfaced as an answer failure), and a 2xx response that is not a well-formed
completion is a *model error* (also an answer failure, never a search
failure).
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from library_rag.config import Config
from library_rag.llm import (
    AnswerModelError,
    AnswerModelUnavailableError,
    FakeAnswerModel,
    LlamaCppAnswerModel,
    make_answer_model,
)


def _dead_port() -> int:
    """A port nothing is listening on: bound, then released by the close."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _model(**overrides: Any) -> LlamaCppAnswerModel:
    args: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 8080,
        "model_name": "qwen",
        "model_revision": "qwen2.5-7b@abc",
        "timeout_seconds": 1.0,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    args.update(overrides)
    return LlamaCppAnswerModel(**args)


# --- FakeAnswerModel -------------------------------------------------------------


def test_fake_model_defaults_and_records_calls() -> None:
    model = FakeAnswerModel()
    out = model.complete([{"role": "user", "content": "what?"}])
    assert out == "The evidence supports this [E1]."
    assert len(model.calls) == 1
    assert model.calls[0] == [{"role": "user", "content": "what?"}]


def test_fake_model_script_pops_in_order_then_falls_back() -> None:
    model = FakeAnswerModel(["one [E2].", "two [E3]."])
    assert model.complete([{"role": "user", "content": "q1"}]) == "one [E2]."
    assert model.complete([{"role": "user", "content": "q2"}]) == "two [E3]."
    assert model.complete([{"role": "user", "content": "q3"}]) == "The evidence supports this [E1]."
    assert len(model.calls) == 3


# --- error taxonomy ---------------------------------------------------------------


def test_unavailable_is_a_model_error() -> None:
    assert issubclass(AnswerModelUnavailableError, AnswerModelError)


# --- factory -----------------------------------------------------------------------


def test_make_answer_model_unconfigured_is_none(base_config: Config) -> None:
    assert make_answer_model(base_config) is None


def test_make_answer_model_fake(base_config: Config) -> None:
    base_config.answer.fake = True
    assert isinstance(make_answer_model(base_config), FakeAnswerModel)


def test_make_answer_model_configured_builds_client(base_config: Config) -> None:
    base_config.answer.model_revision = "qwen2.5-7b@abc"
    model = make_answer_model(base_config)
    assert isinstance(model, LlamaCppAnswerModel)
    assert model.model_revision == "qwen2.5-7b@abc"


# --- LlamaCppAnswerModel -----------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Stands in for ``httpx.Client``: records the request, canned response."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.url: str | None = None
        self.payload: dict[str, Any] | None = None

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        self.url = url
        self.payload = json
        return self.response


def test_llama_cpp_dead_port_is_unavailable() -> None:
    model = _model(port=_dead_port())
    with pytest.raises(AnswerModelUnavailableError, match="unreachable"):
        model.complete([{"role": "user", "content": "hi"}])


def test_llama_cpp_contracted_payload_and_content(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    client = _FakeClient(
        _FakeResponse(200, {"choices": [{"message": {"content": "The answer [E1]."}}]})
    )
    monkeypatch.setattr(model, "_client", client)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert model.complete(messages) == "The answer [E1]."
    assert client.url == "http://127.0.0.1:8080/v1/chat/completions"
    assert client.payload == {
        "model": "qwen",
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }


def test_llama_cpp_http_error_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    client = _FakeClient(_FakeResponse(500, {}))
    monkeypatch.setattr(model, "_client", client)
    with pytest.raises(AnswerModelUnavailableError, match="HTTP 500"):
        model.complete([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": 42}}]},
        {"no_choices": True},
    ],
    ids=["empty-choices", "non-str-content", "missing-key"],
)
def test_llama_cpp_malformed_body_is_a_permanent_model_error(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    model = _model()
    client = _FakeClient(_FakeResponse(200, payload))
    monkeypatch.setattr(model, "_client", client)
    with pytest.raises(AnswerModelError, match="malformed") as excinfo:
        model.complete([{"role": "user", "content": "hi"}])
    # A malformed body is a model error, not an availability problem.
    assert not isinstance(excinfo.value, AnswerModelUnavailableError)
