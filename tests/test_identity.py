"""Deterministic identities: stable across calls, correct under rename/change."""

from __future__ import annotations

from library_rag.identity import (
    document_id,
    make_task_key,
    normalize_path,
    revision_id,
)


def test_document_id_is_deterministic_and_content_anchored() -> None:
    a = "aa" * 32
    b = "bb" * 32
    assert document_id(a) == document_id(a)
    assert document_id(a) != document_id(b)
    # UUID-format (so usable as a Qdrant point ID).
    assert len(document_id(a).split("-")) == 5


def test_revision_id_is_deterministic_per_doc_and_content() -> None:
    doc = document_id("aa" * 32)
    r1 = revision_id(doc, "cc" * 32)
    assert r1 == revision_id(doc, "cc" * 32)
    # Different content -> different revision under the same document.
    assert r1 != revision_id(doc, "dd" * 32)
    # Same content under a different document -> different revision.
    assert r1 != revision_id(document_id("ee" * 32), "cc" * 32)


def test_normalize_path_is_absolute_and_canonical() -> None:
    # Collapses ./ and // and is absolute.
    assert normalize_path("/books/./a/../b.pdf") == "/books/b.pdf"
    assert normalize_path("/x/y//z.epub").endswith("/x/y/z.epub")


def test_task_key_is_a_pure_function_of_stage_input_version_range() -> None:
    k1 = make_task_key("extract", "doc1", "cfg-v1", "pages:0-9")
    k2 = make_task_key("extract", "doc1", "cfg-v1", "pages:0-9")
    assert k1 == k2
    # Any change to a defining component changes the key.
    assert k1 != make_task_key("extract", "doc1", "cfg-v2", "pages:0-9")
    assert k1 != make_task_key("extract", "doc2", "cfg-v1", "pages:0-9")
    assert k1 != make_task_key("extract", "doc1", "cfg-v1", "pages:10-19")
    assert k1 != make_task_key("chunk", "doc1", "cfg-v1", "pages:0-9")
