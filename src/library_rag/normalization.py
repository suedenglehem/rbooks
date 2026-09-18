"""Searchable-text normalization with source-span maps (PRD §8E).

The searchable text and the source text are kept *separate*: normalization
(collapse whitespace, dehyphenate, drop repeated headers/footers) rewrites a
compact searchable string, and every searchable character carries the exact
``[source_start, source_end)`` interval of the source text it came from. That
is what makes every quotation quotable back to the source: a quote is always
``source[span]`` of the *original* extracted text, never the normalized text.

Rules (all configurable via :class:`NormalizationSettings`):

* a whitespace run with no newline collapses to ``" "``; a run with two or
  more newlines to ``"\n"`` (a paragraph break); a run with exactly one
  newline to ``"\n "`` when more whitespace follows the newline in the run, to
  ``"\n"`` at the start/end of the unit, right after a kept hyphen, or right
  before removed running matter, and to ``" "`` otherwise (an ordinary line
  break inside a paragraph); in every case the run's span covers the whole
  run, so the quote includes the original spacing;
* a line-end hyphen joins the two fragments (``"de-" + newline + "line"`` ->
  ``"deline"``). The hyphen and the whitespace are dropped; the characters of
  both fragments keep their own 1:1 spans. A double newline (paragraph break)
  never dehyphenates — that is what makes the rule safe on EPUB sections;
* a repeated first line (header) or last line (footer) is removed only when it
  recurs on at least ``header_min_pages`` units of the same book, has a
  stripped length within ``[header_min_chars, header_max_chars]``, and only for
  units where cross-page repetition is meaningful (PDF pages, never EPUB
  sections — a section's first line is its title or content, not running
  matter).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import NormalizationSettings

__all__ = ["NormalizedUnit", "normalize_unit", "normalize_units", "unit_removed_ranges"]


@dataclass(frozen=True)
class NormalizedUnit:
    """A unit's searchable text plus the per-character span map to the source."""

    searchable: str
    # One [source_start, source_end) interval per searchable character.
    spans: list[list[int]]


def _lines(text: str) -> list[tuple[int, int, str]]:
    """(start, end, stripped) for each physical line; end includes the newline."""
    out: list[tuple[int, int, str]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            out.append((start, i + 1, text[start:i].strip()))
            start = i + 1
    if start < len(text):
        out.append((start, len(text), text[start:].strip()))
    return out


def _normalize_chars(
    text: str, settings: NormalizationSettings, removed: list[tuple[int, int]]
) -> tuple[str, list[list[int]]]:
    """Single char pass: collapse, dehyphenate, and skip removed ranges."""
    out_chars: list[str] = []
    spans: list[list[int]] = []
    n = len(text)
    i = 0
    ri = 0  # index of the removed range we are inside / past
    while i < n:
        while ri < len(removed) and i >= removed[ri][1]:
            ri += 1
        if ri < len(removed) and removed[ri][0] <= i < removed[ri][1]:
            i = removed[ri][1]
            continue
        c = text[i]
        if c.isspace():
            if not settings.collapse_whitespace:
                out_chars.append(c)
                spans.append([i, i + 1])
                i += 1
                continue
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            run = text[i:j]
            if "\n" not in run:
                collapsed = " "
            elif run.count("\n") >= 2:
                collapsed = "\n"  # paragraph break
            else:
                last_nl = i + run.rindex("\n")
                if j > last_nl + 1:
                    collapsed = "\n "  # newline plus trailing run whitespace
                elif i == 0 or text[i - 1] == "-" or j == n:
                    collapsed = "\n"  # unit edge, kept hyphen, or now-trailing
                elif ri < len(removed) and removed[ri][0] <= j < removed[ri][1]:
                    collapsed = "\n"  # newline before removed running matter
                else:
                    collapsed = " "  # ordinary line break within a paragraph
            out_chars.append(collapsed)
            # "\n " is two searchable characters; every one of them maps to the
            # same source whitespace run.
            spans.extend([[i, j]] * len(collapsed))
            i = j
            continue
        # Dehyphenate "word-\nword": alpha before '-', a single newline in the
        # following whitespace run, a lowercase letter after it.
        if (
            settings.dehyphenate
            and settings.collapse_whitespace
            and c == "-"
            and i > 0
            and text[i - 1].isalpha()
        ):
            k = i + 1
            while k < n and text[k] in " \t":
                k += 1
            if k < n and text[k] == "\n":
                k += 1
                after = text[k] if k < n else ""
                if (k >= n or text[k] != "\n") and after and after.islower():
                    i = k  # drop '-' through the whitespace run
                    continue
        out_chars.append(c)
        spans.append([i, i + 1])
        i += 1
    return "".join(out_chars), spans


def _qualifies(stripped: str, settings: NormalizationSettings) -> bool:
    return settings.header_min_chars <= len(stripped) <= settings.header_max_chars


def unit_removed_ranges(
    unit_texts: list[str],
    settings: NormalizationSettings,
    cross_unit: bool = True,
) -> list[list[tuple[int, int]]]:
    """Per-unit source ranges to remove (running headers / footers).

    A first line becomes a header only when its stripped form occurs as a
    first line on at least ``header_min_pages`` units (and fits the length
    bounds); footers work the same on last lines. With ``cross_unit=False``
    (EPUB sections) nothing is removed: repetition across sections is not
    running matter.
    """
    n = len(unit_texts)
    ranges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    if not (settings.remove_repeated_headers and cross_unit and n >= 2):
        return ranges

    lines = [_lines(t) for t in unit_texts]
    first_counts: dict[str, int] = {}
    last_counts: dict[str, int] = {}
    for ls in lines:
        if not ls:
            continue
        fl = ls[0][2]
        ll = ls[-1][2]
        if fl and _qualifies(fl, settings):
            first_counts[fl] = first_counts.get(fl, 0) + 1
        if ll and _qualifies(ll, settings):
            last_counts[ll] = last_counts.get(ll, 0) + 1
    headers = {s for s, c in first_counts.items() if c >= settings.header_min_pages}
    footers = {s for s, c in last_counts.items() if c >= settings.header_min_pages}

    for idx, ls in enumerate(lines):
        ur: list[tuple[int, int]] = []
        if ls:
            fl_s, fl_e, fl = ls[0]
            ll_s, ll_e, ll = ls[-1]
            if fl in headers:
                ur.append((fl_s, fl_e))
            if ll in footers:
                ur.append((ll_s, ll_e))
            ur.sort()
            merged: list[tuple[int, int]] = []
            for s, e in ur:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            ranges[idx] = merged
    return ranges


def normalize_unit(
    text: str, removed: list[tuple[int, int]], settings: NormalizationSettings
) -> NormalizedUnit:
    searchable, spans = _normalize_chars(text, settings, sorted(removed))
    return NormalizedUnit(searchable=searchable, spans=spans)


def normalize_units(
    unit_texts: list[str], settings: NormalizationSettings, cross_unit: bool = True
) -> list[NormalizedUnit]:
    """Normalize a position-ordered list of unit source texts together."""
    ranges = unit_removed_ranges(unit_texts, settings, cross_unit=cross_unit)
    return [normalize_unit(t, r, settings) for t, r in zip(unit_texts, ranges, strict=True)]
