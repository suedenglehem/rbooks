"""Token-budget chunking (PRD §8E): the no-truncation and progress invariants,
line-end cut preference, cross-unit citation spans, OCR bbox unions, the EPUB
title prefix, and deterministic chunk ids.

Units here carry identity span maps (``[[i, i + 1] ...]``): span-mapping
correctness is normalized-text's job (see test_normalization.py); what matters
for chunking is that a token's span is the first-to-last character interval of
its source run.
"""

from __future__ import annotations

from collections.abc import Sequence

from library_rag.chunking import UnitInput, chunk_units
from library_rag.config import ChunkingSettings


def _unit(
    unit_id: str,
    text: str,
    title: str | None = None,
    word_boxes: Sequence[tuple[str, list[float]]] = (),
) -> UnitInput:
    return UnitInput(
        unit_id=unit_id,
        searchable=text,
        spans=[[i, i + 1] for i in range(len(text))],
        title=title,
        word_boxes=tuple(word_boxes),
    )


def test_no_token_is_truncated() -> None:  # M3 gate: no truncation
    words = [f"w{i:02d}" for i in range(50)]
    s = ChunkingSettings(target_tokens=10, overlap_tokens=2, max_tokens=50)
    chunks = chunk_units([_unit("u", " ".join(words))], s, "run-1")

    assert len(chunks) > 1
    # Every source token lands in at least one chunk.
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(words)
    # Each chunk's tokens are one consecutive slice of the source, and the
    # slices start strictly later each time (progress).
    starts: list[int] = []
    for c in chunks:
        toks = c.text.split()
        idx = [words.index(t) for t in toks]
        assert idx == list(range(idx[0], idx[0] + len(idx)))
        starts.append(idx[0])
        assert c.token_count <= 10
    assert starts == sorted(set(starts))
    # The final token ends the final chunk.
    assert chunks[-1].text.split()[-1] == "w49"


def test_cut_prefers_line_end() -> None:
    text = " ".join(f"a{i}" for i in range(9)) + "\nb0 b1"
    s = ChunkingSettings(target_tokens=10, overlap_tokens=2, max_tokens=50)
    chunks = chunk_units([_unit("u", text)], s, "run-1")
    # Budget is 10, but the cut lands at the line end: the first chunk ends
    # with a8, and the second chunk resumes in the overlap.
    assert chunks[0].text == "a0 a1 a2 a3 a4 a5 a6 a7 a8"
    assert chunks[1].text == "a7 a8 b0 b1"


def test_cut_falls_back_to_budget_without_line_end() -> None:
    words = [f"w{i:02d}" for i in range(15)]
    s = ChunkingSettings(target_tokens=10, overlap_tokens=0, max_tokens=50)
    chunks = chunk_units([_unit("u", " ".join(words))], s, "run-1")
    assert [c.text for c in chunks] == [
        " ".join(words[:10]),
        " ".join(words[10:]),
    ]


def test_spans_are_maximal_same_unit_runs() -> None:
    s = ChunkingSettings(target_tokens=10, overlap_tokens=0, max_tokens=50)
    chunks = chunk_units([_unit("u0", "aa bb"), _unit("u1", "cc dd")], s, "run-1")
    assert len(chunks) == 1
    assert chunks[0].text == "aa bb cc dd"
    assert [(sp.unit_id, sp.source_start, sp.source_end) for sp in chunks[0].spans] == [
        ("u0", 0, 5),
        ("u1", 0, 5),
    ]
    assert all(sp.bbox is None for sp in chunks[0].spans)  # native text: no boxes


def test_span_bbox_is_union_of_overlapping_word_boxes() -> None:
    unit = _unit("u", "aa bb", word_boxes=[("aa", [10, 20, 5, 5]), ("bb", [30, 25, 5, 5])])
    s = ChunkingSettings(target_tokens=10, overlap_tokens=0, max_tokens=50)
    chunks = chunk_units([unit], s, "run-1")
    assert len(chunks) == 1
    assert chunks[0].spans[0].bbox == [10.0, 20.0, 35.0, 30.0]


def test_partial_overlap_counts_only_overlapping_boxes() -> None:
    # target 2: "aa bb" / "cc"; cc's box must not leak into the first span.
    unit = _unit(
        "u",
        "aa bb cc",
        word_boxes=[("aa", [0, 0, 1, 1]), ("bb", [10, 0, 1, 1]), ("cc", [20, 0, 1, 1])],
    )
    s = ChunkingSettings(target_tokens=2, overlap_tokens=0, max_tokens=50)
    chunks = chunk_units([unit], s, "run-1")
    assert [c.text for c in chunks] == ["aa bb", "cc"]
    assert chunks[0].spans[0].bbox == [0.0, 0.0, 11.0, 1.0]
    assert chunks[1].spans[0].bbox == [20.0, 0.0, 21.0, 1.0]
    # The second chunk's span starts after the first one ended.
    assert chunks[1].spans[0].source_start > chunks[0].spans[0].source_end


def test_title_prefix_attached_to_first_chunk_only() -> None:
    s = ChunkingSettings(target_tokens=6, overlap_tokens=0, max_tokens=50)
    chunks = chunk_units(
        [_unit("u", "one two three four", title="Chapter One")], s, "run-1"
    )
    assert len(chunks) == 1
    assert chunks[0].title == "Chapter One"
    # Title words count against the budget but never enter the chunk text.
    assert chunks[0].token_count == 6  # 4 body + 2 title tokens
    assert chunks[0].text == "one two three four"
    assert not any(w in chunks[0].text.split() for w in ("Chapter", "One"))


def test_title_consumes_budget() -> None:
    s = ChunkingSettings(target_tokens=8, overlap_tokens=0, max_tokens=50)
    chunks = chunk_units(
        [_unit("u", "one two three four five", title="A Very Long Chapter Title")],
        s,
        "run-1",
    )
    # 5 title tokens leave a 3-token body budget for the first chunk; the
    # continuation chunk carries no title.
    assert [(c.text, c.title, c.token_count) for c in chunks] == [
        ("one two three", "A Very Long Chapter Title", 8),
        ("four five", None, 2),
    ]


def test_title_prefix_can_be_disabled() -> None:
    s = ChunkingSettings(target_tokens=6, overlap_tokens=0, max_tokens=50, title_prefix=False)
    chunks = chunk_units([_unit("u", "one two three four", title="Chapter One")], s, "run-1")
    assert chunks[0].title is None
    assert chunks[0].token_count == 4


def test_chunk_ids_are_deterministic_and_run_scoped() -> None:
    s = ChunkingSettings(target_tokens=5, overlap_tokens=1, max_tokens=50)
    units = [_unit("u0", "aa bb cc"), _unit("u1", "dd ee ff")]
    a = chunk_units(units, s, "run-1")
    b = chunk_units(units, s, "run-1")
    other = chunk_units(units, s, "run-2")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len(a) > 0
    assert [c.chunk_id for c in a] != [c.chunk_id for c in other]


def test_overlap_still_makes_progress() -> None:
    words = [f"w{i:02d}" for i in range(8)]
    s = ChunkingSettings(target_tokens=5, overlap_tokens=4, max_tokens=50)
    chunks = chunk_units([_unit("u", " ".join(words))], s, "run-1")
    starts = [words.index(c.text.split()[0]) for c in chunks]
    assert starts == sorted(set(starts))  # strictly increasing
    assert chunks[-1].text.split()[-1] == "w07"  # the last token is always reached
    assert all(c.token_count <= 5 for c in chunks)


def test_empty_units_yield_no_chunks() -> None:
    s = ChunkingSettings()
    assert chunk_units([], s, "run-1") == []
    assert chunk_units([_unit("u", "   \n  ")], s, "run-1") == []
