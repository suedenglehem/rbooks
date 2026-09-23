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
import threading
from collections.abc import Sequence
from typing import Any

import pytest

from library_rag.config import AnswerEndpoint, Config
from library_rag.llm import (
    AnswerModelError,
    AnswerModelPool,
    AnswerModelUnavailableError,
    FakeAnswerModel,
    LlamaCppAnswerModel,
    Message,
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


def test_make_answer_model_without_extra_endpoint_stays_bare(base_config: Config) -> None:
    base_config.answer.model_revision = "qwen2.5-7b@abc"
    model = make_answer_model(base_config)
    assert isinstance(model, LlamaCppAnswerModel)
    assert not isinstance(model, AnswerModelPool)


def test_make_answer_model_with_extra_endpoint_builds_pool(base_config: Config) -> None:
    base_config.answer.model_revision = "qwen2.5-7b@abc"
    base_config.answer.extra_endpoints = [AnswerEndpoint(host="ak", port=8080)]
    model = make_answer_model(base_config)
    assert isinstance(model, AnswerModelPool)
    assert len(model) == 2
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


def test_llama_cpp_null_content_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A thinking model (qwen3.8-27b) that spends max_tokens inside the
    # reasoning trace returns 200 with content: null. That is a per-call
    # budget condition, not a broken server: it must be *unavailable*
    # (transient → job retries with backoff, pool can fail over), not a
    # permanent malformed-response failure. Regression: 362.PDF burned its
    # two retry attempts on exactly this before the backfill finished.
    model = _model()
    client = _FakeClient(
        _FakeResponse(200, {"choices": [{"message": {"content": None}}]})
    )
    monkeypatch.setattr(model, "_client", client)
    with pytest.raises(AnswerModelUnavailableError, match="null content"):
        model.complete([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("content", ["", "   \n  "], ids=["empty", "whitespace"])
def test_llama_cpp_empty_content_is_transient(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    # Same budget condition as null content, different server: llama.cpp
    # (ak) returns 200 with content: "" when the reasoning trace eats the
    # max_tokens budget, where vLLM returns null. Must be *unavailable*
    # (transient -> retry + pool failover), never a permanent failure.
    # Regression: 2026-09-23 dual-LLM regen — every resume job routed to
    # ak permanent-failed with resume_too_short (0 words) under the old
    # check, because "" passed the isinstance(str) gate.
    model = _model()
    client = _FakeClient(
        _FakeResponse(200, {"choices": [{"message": {"content": content}}]})
    )
    monkeypatch.setattr(model, "_client", client)
    with pytest.raises(AnswerModelUnavailableError, match="empty content"):
        model.complete([{"role": "user", "content": "hi"}])


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


# --- AnswerModelPool (M9) ----------------------------------------------------------

_MSGS: list[dict[str, str]] = [{"role": "user", "content": "hi"}]


class _FailingModel:
    """Always raises the given error; counts the calls it was asked to make."""

    model_revision = "failing-v1"

    def __init__(self, exc: AnswerModelError) -> None:
        self._exc = exc
        self.calls = 0

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls += 1
        raise self._exc


class _GatedModel:
    """Blocks inside ``complete()`` until released; signals on arrival.

    ``arrived`` is set once the call has started — after the pool counted the
    endpoint in-flight — and ``release`` gates the return. Holding an endpoint
    busy this way lets a test observe where a concurrent call lands, with no
    timing dependence.
    """

    model_revision = "gated-v1"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.arrived = threading.Event()
        self.release = threading.Event()

    def complete(self, messages: Sequence[Message]) -> str:
        self.arrived.set()
        self.release.wait()
        return self._reply


def test_pool_round_robin_between_endpoints() -> None:
    # Idle tie-break: with nothing in flight the ratios are all equal, so
    # sequential calls still alternate endpoints round-robin (M9 behavior).
    a = FakeAnswerModel(["a1", "a2"])
    b = FakeAnswerModel(["b1", "b2"])
    pool = AnswerModelPool([a, b])
    assert [pool.complete(_MSGS) for _ in range(4)] == ["a1", "b1", "a2", "b2"]


def test_pool_least_inflight_prefers_free_endpoint() -> None:
    a = _GatedModel("from-a")
    b = _GatedModel("from-b")
    pool = AnswerModelPool([a, b])
    results: dict[str, str] = {}

    def call(tag: str) -> None:
        results[tag] = pool.complete(_MSGS)

    t1 = threading.Thread(target=call, args=("t1",))
    t1.start()
    a.arrived.wait()  # t1 takes `a` (idle tie -> round-robin position 0)
    t2 = threading.Thread(target=call, args=("t2",))
    t2.start()
    # `a` is in flight and `b` is free, so the concurrent call MUST land on
    # `b` — this is what lets the faster endpoint carry a bigger share.
    b.arrived.wait()
    a.release.set()
    b.release.set()
    t1.join()
    t2.join()
    assert results == {"t1": "from-a", "t2": "from-b"}
    assert pool.inflight() == [0, 0]


def test_pool_weights_tilt_a_loaded_tie() -> None:
    # Both endpoints busy: the ratio is inflight/weight, so the weight-2
    # endpoint (1/2 = 0.5) beats weight 1 (1/1 = 1.0) and gets the next call.
    a = _GatedModel("from-a")
    b = _GatedModel("from-b")
    pool = AnswerModelPool([a, b], weights=[2, 1])
    results: dict[str, str] = {}

    def call(tag: str) -> None:
        results[tag] = pool.complete(_MSGS)

    t1 = threading.Thread(target=call, args=("t1",))
    t1.start()
    a.arrived.wait()  # t1 takes `a` (idle tie -> round-robin position 0)
    t2 = threading.Thread(target=call, args=("t2",))
    t2.start()
    b.arrived.wait()  # `a` busy -> `b` is the only free endpoint
    t3 = threading.Thread(target=call, args=("t3",))
    t3.start()
    a.release.set()
    b.release.set()
    t1.join()
    t2.join()
    t3.join()
    assert results == {"t1": "from-a", "t2": "from-b", "t3": "from-a"}
    assert pool.inflight() == [0, 0]


def test_pool_inflight_released_after_failover() -> None:
    down = _FailingModel(AnswerModelUnavailableError("connection refused"))
    up = FakeAnswerModel(["ok [E1]."])
    pool = AnswerModelPool([down, up])
    assert pool.complete(_MSGS) == "ok [E1]."
    # The failed attempt on `down` and the successful one on `up` both
    # released their counters — no phantom in-flight calls accumulate.
    assert pool.inflight() == [0, 0]


def test_pool_inflight_released_on_unexpected_error() -> None:
    class _BlowingModel:
        model_revision = "blowing-v1"

        def complete(self, messages: Sequence[Message]) -> str:
            raise RuntimeError("boom")

    up = FakeAnswerModel()
    pool = AnswerModelPool([_BlowingModel(), up])
    with pytest.raises(RuntimeError, match="boom"):
        pool.complete(_MSGS)
    # A non-AnswerModelError is a bug, not a dead endpoint: it propagates
    # without failover, but the counter still comes back down.
    assert pool.inflight() == [0, 0]
    assert up.calls == []


def test_pool_rejects_bad_weights() -> None:
    a = FakeAnswerModel()
    b = FakeAnswerModel()
    with pytest.raises(ValueError, match="one entry per model"):
        AnswerModelPool([a, b], weights=[1])
    with pytest.raises(ValueError, match=">= 1"):
        AnswerModelPool([a, b], weights=[2, 0])


def test_pool_exposes_len_and_first_model_revision() -> None:
    a = FakeAnswerModel()
    b = FakeAnswerModel()
    pool = AnswerModelPool([a, b])
    assert len(pool) == 2
    assert pool.model_revision == a.model_revision


def test_pool_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AnswerModelPool([])


def test_pool_fails_over_when_an_endpoint_is_unavailable() -> None:
    down = _FailingModel(AnswerModelUnavailableError("connection refused"))
    # Two scripted replies: call 1 reaches `up` via failover (off the dead
    # endpoint), and call 2 lands on `up` directly because the idle
    # tie-break's round-robin position already advanced past the dead
    # endpoint on call 1.
    up = FakeAnswerModel(["ok [E1].", "ok [E1]."])
    pool = AnswerModelPool([down, up])
    assert pool.complete(_MSGS) == "ok [E1]."
    assert down.calls == 1
    assert pool.complete(_MSGS) == "ok [E1]."


def test_pool_fails_over_on_malformed_responses_too() -> None:
    broken = _FailingModel(AnswerModelError("malformed chat completion response"))
    up = FakeAnswerModel(["ok [E1]."])
    pool = AnswerModelPool([broken, up])
    assert pool.complete(_MSGS) == "ok [E1]."
    assert broken.calls == 1


def test_pool_raises_last_error_when_every_endpoint_fails() -> None:
    e1 = _FailingModel(AnswerModelUnavailableError("first down"))
    e2 = _FailingModel(AnswerModelError("second broken"))
    pool = AnswerModelPool([e1, e2])
    with pytest.raises(AnswerModelError, match="second broken"):
        pool.complete(_MSGS)
    assert e1.calls == 1
    assert e2.calls == 1
