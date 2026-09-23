"""Local answer-model adapter (PRD §12).

Mirrors the embedding adapter: a small :class:`AnswerModel` protocol, a
llama.cpp server client (OpenAI-compatible ``/v1/chat/completions``) with
bounded timeouts, and a deterministic test double for the unit suite.

The model is a pure text transducer: it receives message lists (system +
user, plus the bounded repair turn) and returns raw text. All citation
validation, abstention parsing, and the never-fabricate rule live in
``citations.py``/``answers.py`` — this module only distinguishes *the server
is unavailable* (transient; search and other commands must keep working) from
*a permanent, malformed response*.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from .config import Config

__all__ = [
    "AnswerModel",
    "AnswerModelError",
    "AnswerModelPool",
    "AnswerModelUnavailableError",
    "FakeAnswerModel",
    "LlamaCppAnswerModel",
    "Message",
    "build_answer_pool",
    "make_answer_model",
]

# One chat message: {"role": "system" | "user" | "assistant", "content": str}.
Message = dict[str, str]


class AnswerModelError(RuntimeError):
    """A permanent answer-model failure (malformed server response)."""


class AnswerModelUnavailableError(AnswerModelError):
    """The answer-model server is unreachable or timed out (transient).

    The caller reports the failure explicitly and shows the evidence it
    already has; it never fabricates an answer. Retrieval (``search``) does
    not use this adapter at all and is unaffected (PRD §12 gate).
    """


@runtime_checkable
class AnswerModel(Protocol):
    """Completes a chat conversation with one bounded text response."""

    model_revision: str

    def complete(self, messages: Sequence[Message]) -> str: ...


class FakeAnswerModel:
    """Deterministic answer-model double for tests.

    Without scripted replies it answers a generic sentence citing ``E1`` —
    the happy path every gate test needs. Scripted replies are popped in
    order (scripted repair turns included); once the script is exhausted the
    default answer is used again. ``calls`` records every message list so
    tests can assert the exact number of model invocations (the one-repair
    bound) and their content.
    """

    model_revision = "fake-answer-v1"

    _DEFAULT = "The evidence supports this [E1]."

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self._replies: list[str] = list(replies) if replies is not None else []
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        if self._replies:
            return self._replies.pop(0)
        return self._DEFAULT


class LlamaCppAnswerModel:
    """llama.cpp server client (OpenAI-compatible chat completions).

    Bounded by ``timeout_seconds`` end-to-end (connect gets its own, shorter
    budget). Network-level failures and non-2xx responses are
    :class:`AnswerModelUnavailableError`; a 2xx response that is not
    well-formed JSON with a text choice is :class:`AnswerModelError`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        *,
        model_revision: str,
        timeout_seconds: float,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.model_revision = model_revision
        self._url = f"http://{host}:{port}/v1/chat/completions"
        self._model = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        # The connect phase fails fast on a dead loopback port; the read
        # phase is where generation time actually lives.
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10.0))

    def complete(self, messages: Sequence[Message]) -> str:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        try:
            resp = self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise AnswerModelUnavailableError(
                f"answer model server unreachable at {self._url}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise AnswerModelUnavailableError(
                f"answer model server returned HTTP {resp.status_code}"
            )
        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise AnswerModelError(
                f"malformed chat completion response: {exc}"
            ) from exc
        if content is None:
            # Thinking models (qwen3.8-27b) that spend max_tokens inside the
            # reasoning trace return 200 with content: null. That is a
            # per-call budget condition, not a broken server — transient, so
            # the job retries with backoff and the pool can fail over.
            raise AnswerModelUnavailableError(
                "model returned null content (thinking budget exhausted?)"
            )
        if not isinstance(content, str):
            raise AnswerModelError(
                "malformed chat completion response: content is not a string"
            )
        if not content.strip():
            # The same budget condition as null content, from a different
            # server: llama.cpp returns content: "" when the reasoning trace
            # eats the max_tokens budget (vLLM returns null instead).
            # Transient -> retry + failover, never a permanent model error.
            raise AnswerModelUnavailableError(
                "model returned empty content (thinking budget exhausted?)"
            )
        return content


class AnswerModelPool:
    """A load-balancing pool of endpoints serving the same model (M9).

    Calls start at the *least-loaded* endpoint — the one with the fewest
    in-flight calls per unit of weight (``inflight_i / weight_i``, compared
    by cross-multiplication) — and, on :class:`AnswerModelError` (including
    :class:`AnswerModelUnavailableError`), fail over to the next endpoint
    for the *same* call. So an endpoint that is down costs its (fast)
    connect timeout per affected call and the pool degrades to the live
    ones; if every endpoint fails, the last error is raised.

    Weights (default 1 for every endpoint) are relative-capacity
    multipliers on top of each endpoint's observed speed: with all weights
    1, in steady state dispatch is proportional to speed (the fast
    endpoint's in-flight count drains faster, so it keeps winning). Raising
    an endpoint's weight routes it a share *beyond* its natural speed
    share; the useful direction is raising the primary's weight to shed
    batch load off a shared secondary (e.g. a machine also serving
    interactive traffic).

    Idle ties break round-robin: when the ratios are equal (in particular
    when nothing is in flight) the next round-robin position wins. That
    keeps the pool's cold-start and idle behavior identical to the
    original M9 round-robin — including only probing a down endpoint
    every other call — and lets least-inflight take over once concurrency
    exceeds the endpoint count.

    Thread-safe: the worker may issue completions from several job threads
    (``worker.max_concurrent_jobs``), and each ``httpx.Client`` inside a
    :class:`LlamaCppAnswerModel` is itself safe to share across threads.

    A single-element pool is never constructed — :func:`build_answer_pool`
    returns the bare model instead, so the no-extra-endpoints path is
    byte-identical to the pre-M9 behavior.
    """

    def __init__(
        self, models: Sequence[AnswerModel], weights: Sequence[int] | None = None
    ) -> None:
        if not models:
            raise ValueError("AnswerModelPool requires at least one model")
        self._models = list(models)
        n = len(self._models)
        if weights is None:
            weights = (1,) * n
        if len(weights) != n:
            raise ValueError("weights must have one entry per model")
        if any(w < 1 for w in weights):
            raise ValueError("weights must be >= 1")
        self._weights = list(weights)
        self._lock = threading.Lock()
        self._rr = 0
        self._inflight = [0] * n
        self.model_revision = self._models[0].model_revision

    def __len__(self) -> int:
        return len(self._models)

    def inflight(self) -> list[int]:
        """A snapshot of per-endpoint in-flight call counts (diagnostics)."""
        with self._lock:
            return list(self._inflight)

    def _pick_locked(self) -> int:
        """Least in-flight-per-weight endpoint; round-robin breaks ties.

        The caller holds ``self._lock``. Comparison is
        ``inflight_i * w_best < inflight_best * w_i`` (strict), so equal
        ratios keep the earliest candidate in round-robin order.
        """
        n = len(self._models)
        best = self._rr
        self._rr = (self._rr + 1) % n
        for offset in range(1, n):
            i = (best + offset) % n
            if self._inflight[i] * self._weights[best] < self._inflight[best] * self._weights[i]:
                best = i
        return best

    def complete(self, messages: Sequence[Message]) -> str:
        n = len(self._models)
        with self._lock:
            i = self._pick_locked()
        last_error: AnswerModelError | None = None
        for _ in range(n):
            with self._lock:
                self._inflight[i] += 1
            try:
                return self._models[i].complete(messages)
            except AnswerModelError as exc:
                last_error = exc
            finally:
                with self._lock:
                    self._inflight[i] -= 1
            i = (i + 1) % n  # reached only when the attempt failed
        assert last_error is not None  # the loop ran n >= 1 times
        raise last_error


def build_answer_pool(
    cfg: Config,
    *,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
) -> AnswerModel | None:
    """Build the answer model: primary endpoint, plus a pool when configured.

    Unlike :func:`library_rag.embeddings.make_embedder`, this never raises:
    the answer model is optional (search and the reader must keep working
    without it), and an unconfigured model is reported as a failed answer
    with an explicit reason, not a config crash.

    ``answer.extra_endpoints`` (M9) adds endpoints serving the same model,
    each built with the caller's generation parameters (the resume stage
    passes its own ``max_tokens``/``temperature``/``timeout_seconds``, which
    is why this builder takes them rather than reading ``cfg.answer``).

    Pool weights come from ``answer.weight`` (primary) and each extra
    endpoint's ``weight`` — see :class:`AnswerModelPool` for what they do.
    """
    a = cfg.answer
    if not a.is_configured:
        return None
    if a.fake:
        return FakeAnswerModel()
    assert a.model_revision is not None
    model_name = a.model_name or a.model_revision
    primary = LlamaCppAnswerModel(
        host=cfg.services.answer_host,
        port=cfg.services.answer_port,
        model_name=model_name,
        model_revision=a.model_revision,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not a.extra_endpoints:
        return primary
    models = [primary]
    for e in a.extra_endpoints:
        models.append(
            LlamaCppAnswerModel(
                host=e.host,
                port=e.port,
                model_name=e.model_name or model_name,
                model_revision=a.model_revision,
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    return AnswerModelPool(
        models,
        weights=[a.weight, *(e.weight for e in a.extra_endpoints)],
    )


def make_answer_model(cfg: Config) -> AnswerModel | None:
    """Build the configured answer model, or ``None`` when none is configured.

    Thin wrapper over :func:`build_answer_pool` with the ``answer`` stage's
    generation parameters.
    """
    a = cfg.answer
    return build_answer_pool(
        cfg,
        timeout_seconds=a.timeout_seconds,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
    )
