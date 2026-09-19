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

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from .config import Config

__all__ = [
    "AnswerModel",
    "AnswerModelError",
    "AnswerModelUnavailableError",
    "FakeAnswerModel",
    "LlamaCppAnswerModel",
    "Message",
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
            if not isinstance(content, str):
                raise TypeError("content is not a string")
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise AnswerModelError(
                f"malformed chat completion response: {exc}"
            ) from exc
        return content


def make_answer_model(cfg: Config) -> AnswerModel | None:
    """Build the configured answer model, or ``None`` when none is configured.

    Unlike :func:`library_rag.embeddings.make_embedder`, this never raises:
    the answer model is optional (search and the reader must keep working
    without it), and an unconfigured model is reported as a failed answer
    with an explicit reason, not a config crash.
    """
    a = cfg.answer
    if not a.is_configured:
        return None
    if a.fake:
        return FakeAnswerModel()
    assert a.model_revision is not None
    return LlamaCppAnswerModel(
        host=cfg.services.answer_host,
        port=cfg.services.answer_port,
        model_name=a.model_name or a.model_revision,
        model_revision=a.model_revision,
        timeout_seconds=a.timeout_seconds,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
    )
