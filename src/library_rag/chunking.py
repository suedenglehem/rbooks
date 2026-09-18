"""Token-budget chunking over normalized units (PRD §8E).

Chunks tile the *token stream* (units in position order, each unit's
searchable text tokenized). Invariants:

* **No truncation** — every source token lands in at least one chunk; cuts
  happen only between tokens. The lookback prefers a line end (or a unit
  boundary) within a window of the cut, but if none exists it cuts at the
  budget anyway — always at a token boundary.
* **Progress** — each chunk starts strictly after the previous one, so the
  loop terminates even with a small ``max_tokens`` and a large overlap.
* **Citations** — a chunk's spans are the maximal same-unit runs of its
  tokens, mapped back to *source* character intervals through the unit's span
  map, so a citation always points at source text (OCR or native), never at
  normalized text. For OCR units each span also carries the union bbox of the
  overlapping OCR word boxes (page points) for highlight mapping.
* **Title prefix** — for EPUB sections, the section title is attached to the
  chunk that starts at the section's first token; its tokens count against the
  budget but the title text is excluded from the chunk text and spans (it is
  stored in its own column for display).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

from .config import ChunkingSettings
from .identity import chunk_key

__all__ = [
    "ChunkResult",
    "ChunkSpan",
    "TokenCounter",
    "UnitInput",
    "WordTokenCounter",
    "chunk_units",
]

_WORD = re.compile(r"\S+")


class TokenCounter(Protocol):
    """Tokenizes searchable text into ``(text, start, end)`` tokens."""

    def tokens(self, text: str) -> list[tuple[str, int, int]]: ...


class WordTokenCounter:
    """Deterministic development tokenizer: maximal non-whitespace runs.

    Production BPE counters (e.g. ``tokenizer="bge-m3"``) are resolved lazily
    at chunk time in M4; a missing model is a permanent ``tokenizer_missing``
    failure, never a silent word fallback.
    """

    def tokens(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]


@dataclass(frozen=True)
class UnitInput:
    """One extracted unit as chunking sees it."""

    unit_id: str
    # Normalized searchable text; one [source_start, source_end) per character.
    searchable: str
    spans: list[list[int]]
    title: str | None = None  # EPUB section title (page units: None)
    # OCR word boxes in document order: (word text, page-point box [l, t, w, h]).
    word_boxes: tuple[tuple[str, list[float]], ...] = ()


@dataclass(frozen=True)
class ChunkSpan:
    """A maximal same-unit run of a chunk's tokens, mapped to source text."""

    unit_id: str
    source_start: int
    source_end: int
    bbox: list[float] | None  # union of OCR word boxes, or None for native text


@dataclass(frozen=True)
class ChunkResult:
    """A finished chunk; the worker inserts it with a position and prev/next."""

    chunk_id: str
    text: str
    title: str | None
    token_count: int
    spans: list[ChunkSpan]


@dataclass(frozen=True)
class _Token:
    unit_idx: int
    text: str
    source_start: int
    source_end: int
    preferred: bool  # line end or last token of its unit: preferred cut point
    is_first_of_unit: bool


def _build_tokens(units: list[UnitInput], counter: TokenCounter) -> list[_Token]:
    tokens: list[_Token] = []
    for ui, unit in enumerate(units):
        toks = counter.tokens(unit.searchable)
        for ti, (tok_text, ts, te) in enumerate(toks):
            spans = unit.spans
            preferred = (te < len(unit.searchable) and unit.searchable[te] == "\n") or (
                ti == len(toks) - 1
            )
            tokens.append(
                _Token(
                    unit_idx=ui,
                    text=tok_text,
                    source_start=spans[ts][0],
                    source_end=spans[te - 1][1],
                    preferred=preferred,
                    is_first_of_unit=(ti == 0),
                )
            )
    return tokens


def _word_source_offsets(unit: UnitInput) -> list[tuple[int, int, list[float]]]:
    """(start, end, box) of each OCR word in the searchable text.

    Sequential ``find`` (advancing past each match) tolerates duplicate words;
    a word that normalization changed (e.g. a dehyphenated fragment) is simply
    dropped — losing a box is safe, a wrong box is not.
    """
    text = unit.searchable
    pos = 0
    out: list[tuple[int, int, list[float]]] = []
    for word, box in unit.word_boxes:
        idx = text.find(word, pos)
        if idx == -1:
            continue
        pos = idx + 1
        out.append((idx, idx + len(word), box))
    return out


def _union_word_boxes(unit: UnitInput, s0: int, e1: int) -> list[float] | None:
    rects = [
        box
        for ws, we, box in _word_source_offsets(unit)
        if ws < e1 and we > s0
    ]
    if not rects:
        return None
    return [
        round(min(b[0] for b in rects), 3),
        round(min(b[1] for b in rects), 3),
        round(max(b[0] + b[2] for b in rects), 3),
        round(max(b[1] + b[3] for b in rects), 3),
    ]


def _span_signature(spans: list[ChunkSpan]) -> str:
    pairs = [(s.unit_id, s.source_start, s.source_end) for s in spans]
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()


def _collapse_spans(tokens: list[_Token], units: list[UnitInput]) -> list[ChunkSpan]:
    """Maximal same-unit runs; the token stream is unit-ordered, so at most one
    run per unit per chunk."""
    spans: list[ChunkSpan] = []
    if not tokens:
        return spans
    cur = tokens[0].unit_idx
    s0, e1 = tokens[0].source_start, tokens[0].source_end
    for t in tokens[1:]:
        if t.unit_idx == cur:
            e1 = max(e1, t.source_end)
        else:
            unit = units[cur]
            spans.append(
                ChunkSpan(unit.unit_id, s0, e1, _union_word_boxes(unit, s0, e1))
            )
            cur = t.unit_idx
            s0, e1 = t.source_start, t.source_end
    unit = units[cur]
    spans.append(ChunkSpan(unit.unit_id, s0, e1, _union_word_boxes(unit, s0, e1)))
    return spans


def chunk_units(
    units: list[UnitInput],
    settings: ChunkingSettings,
    run_id: str,
    counter: TokenCounter | None = None,
) -> list[ChunkResult]:
    """Chunk a position-ordered list of units under the token budget."""
    counter = counter or WordTokenCounter()
    tokens = _build_tokens(units, counter)
    n = len(tokens)
    budget = min(settings.target_tokens, settings.max_tokens)
    chunker_sha = settings.settings_sha()
    results: list[ChunkResult] = []
    i = 0
    while i < n:
        title: str | None = None
        title_tokens = 0
        if settings.title_prefix and tokens[i].is_first_of_unit:
            unit_title = units[tokens[i].unit_idx].title
            if unit_title:
                title = unit_title
                title_tokens = len(counter.tokens(unit_title))
        body_budget = max(1, budget - title_tokens)
        c = min(n, i + body_budget)
        if c < n:
            # Prefer a line end / unit boundary within the lookback window;
            # falling back to the raw budget cut is still a token boundary.
            lookback = max(20, settings.target_tokens // 4)
            cut: int | None = None
            for k in range(c - 1, max(i, c - lookback) - 1, -1):
                if tokens[k].preferred:
                    cut = k + 1
                    break
            c = cut if cut is not None else c
        body = tokens[i:c]
        text = " ".join(t.text for t in body)
        spans = _collapse_spans(body, units)
        results.append(
            ChunkResult(
                chunk_id=chunk_key(
                    run_id, chunker_sha, _span_signature(spans),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                ),
                text=text,
                title=title,
                token_count=title_tokens + len(body),
                spans=spans,
            )
        )
        if c >= n:
            break  # chunk reached the end of the stream
        nxt = c - settings.overlap_tokens
        i = nxt if nxt > i else i + 1
    return results
