"""Text normalization (PRD §8E): whitespace collapse, line-end dehyphenation,
and repeated header/footer removal — with the invariant that every searchable
character carries a span back into the *source* text.

``normalize_unit`` is the per-unit entry point (the worker calls it with the
ranges from ``unit_removed_ranges``); every test here exercises it directly so
the span map is checked character by character.
"""

from __future__ import annotations

from library_rag.config import NormalizationSettings
from library_rag.normalization import (
    normalize_unit,
    normalize_units,
    unit_removed_ranges,
)

_S = NormalizationSettings()  # all rules on


def _assert_spans_quote_source(text: str, searchable: str, spans: list[list[int]]) -> None:
    """The core invariant: each searchable char maps to the source chars it came
    from — non-whitespace 1:1, whitespace a non-empty whitespace run."""
    assert len(spans) == len(searchable)
    for ch, (a, b) in zip(searchable, spans, strict=True):
        assert 0 <= a < b <= len(text)
        if ch.isspace():
            assert text[a:b].strip() == ""
        else:
            assert text[a:b] == ch


def test_whitespace_collapse_span_roundtrip() -> None:
    src = "hello \t\n  world"
    unit = normalize_unit(src, [], _S)
    # " \t\n" contains a newline -> "\n"; "  " -> " ".
    assert unit.searchable == "hello\n world"
    _assert_spans_quote_source(src, unit.searchable, unit.spans)


def test_single_newline_is_kept_as_newline() -> None:
    unit = normalize_unit("a  b\n c", [], _S)
    assert unit.searchable == "a b\n c"
    _assert_spans_quote_source("a  b\n c", unit.searchable, unit.spans)


def test_dehyphenation_joins_line_end_fragments() -> None:
    src = "tele-\nphone line"
    unit = normalize_unit(src, [], _S)
    assert unit.searchable == "telephone line"
    _assert_spans_quote_source(src, unit.searchable, unit.spans)
    # The fragment letters map straight to their source positions.
    assert unit.spans[unit.searchable.index("p")] == [6, 7]


def test_dehyphenation_never_crosses_paragraphs_or_capitals() -> None:
    # A paragraph break (double newline) after the hyphen: keep the hyphen.
    unit = normalize_unit("word-\n\nline", [], _S)
    assert unit.searchable == "word-\nline"
    # Capital letter after the break: keep the hyphen.
    assert normalize_unit("word-\nLine", [], _S).searchable == "word-\nLine"
    # Trailing hyphen at end of text: nothing to join with.
    assert normalize_unit("word-", [], _S).searchable == "word-"
    # With the rule disabled the hyphen always stays.
    assert (
        normalize_unit("word-\nline", [], NormalizationSettings(dehyphenate=False)).searchable
        == "word-\nline"
    )


def test_dehyphenation_requires_collapse_whitespace() -> None:
    # Dehyphenation is defined over the collapsed stream; with collapsing off
    # every character is kept 1:1 (hyphen and newline included).
    src = "tele-\nphone"
    unit = normalize_unit(src, [], NormalizationSettings(collapse_whitespace=False))
    assert unit.searchable == src
    assert unit.spans == [[i, i + 1] for i in range(len(src))]


def test_header_removed_when_repeated_across_units() -> None:
    units = normalize_units(["The Tale\nbody one", "The Tale\nbody two", "Other\nbody three"], _S)
    # "The Tale" recurs on the first line of two units -> running header.
    assert units[0].searchable == "body one"
    assert units[1].searchable == "body two"
    # "Other" occurs once -> kept.
    assert units[2].searchable == "Other body three"
    _assert_spans_quote_source("The Tale\nbody one", "body one", units[0].spans)


def test_footer_removed_when_repeated_across_units() -> None:
    units = normalize_units(["body one\nfoot", "body two\nfoot"], _S)
    # The footer line itself is dropped; the preceding newline stays as the
    # (now trailing) paragraph break.
    assert units[0].searchable == "body one\n"
    assert units[1].searchable == "body two\n"


def test_single_occurrence_lines_are_kept() -> None:
    units = normalize_units(["Only once\nalpha", "Different\nbeta"], _S)
    assert units[0].searchable == "Only once alpha"
    assert units[1].searchable == "Different beta"


def test_lines_outside_length_bounds_are_kept() -> None:
    # "7" is shorter than header_min_chars (3): a page number, not a header.
    units = normalize_units(["7\nalpha", "7\nbeta"], _S)
    assert units[0].searchable == "7 alpha"
    # A 90-char line is longer than header_max_chars (80): body text.
    long = "x" * 90
    units = normalize_units([f"{long}\na", f"{long}\nb"], _S)
    assert units[0].searchable == f"{long} a"


def test_removal_needs_two_units() -> None:
    # With a single unit nothing can "recur".
    units = normalize_units(["The Tale\nbody"], _S)
    assert units[0].searchable == "The Tale body"


def test_cross_unit_false_never_removes() -> None:
    units = normalize_units(["The Tale\na", "The Tale\nb"], _S, cross_unit=False)
    assert units[0].searchable == "The Tale a"
    assert units[1].searchable == "The Tale b"


def test_header_removal_can_be_disabled() -> None:
    s = NormalizationSettings(remove_repeated_headers=False)
    units = normalize_units(["The Tale\na", "The Tale\nb"], s)
    assert units[0].searchable == "The Tale a"


def test_unit_removed_ranges_direct() -> None:
    # Header "Head" on units 0,1; footer "Tail" on units 0,1; unit 2 keeps both
    # "Nope" (once) and its single last line.
    ranges = unit_removed_ranges(["Head\nmid\nTail", "Head\nx\nTail", "Nope\ny"], _S)
    assert ranges[0] == [(0, 5), (9, 13)]
    assert ranges[1] == [(0, 5), (7, 11)]
    assert ranges[2] == []
    # normalize_unit drops exactly the removed text and maps what remains.
    unit = normalize_unit("Head\nmid\nTail", ranges[0], _S)
    assert unit.searchable == "mid\n"
    _assert_spans_quote_source("Head\nmid\nTail", unit.searchable, unit.spans)


def test_removed_ranges_survive_normalization() -> None:
    # The kept characters still point into the original source, not a
    # re-based string.
    unit = normalize_unit("Head\nmid", unit_removed_ranges(["Head\nmid", "Head\nx"], _S)[0], _S)
    assert unit.searchable == "mid"
    assert unit.spans == [[5, 6], [6, 7], [7, 8]]
