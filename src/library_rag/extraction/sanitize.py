"""XHTML sanitization for EPUB content (XSS defense, PRD §8D/§12).

EPUB chapter bodies are untrusted markup: they may contain ``<script>``,
event-handler attributes, ``javascript:``/``data:`` URIs, or imports of
external resources. We strip all of it with the standard library
(:mod:`xml.etree.ElementTree`), keeping plain content, element ids, and
in-document ``#fragment`` links (needed for paragraph anchors).

The sanitizer operates on the chapter's XHTML bytes and returns the
*body* of the document as a sanitized XHTML fragment string.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

__all__ = ["XHTML_NS", "parse_blocks", "sanitize"]

XHTML_NS = "http://www.w3.org/1999/xhtml"

# Elements removed entirely (they carry behavior or external fetches).
_FORBIDDEN_TAGS = frozenset(
    {"script", "style", "link", "meta", "iframe", "object", "embed", "form", "base", "template"}
)
# URI attributes that may smuggle in active/external content.
_URI_ATTRS = ("href", "src", "action", "poster", "data")
_EXTERNAL_URI = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")

# Block-level elements that form the paragraph units for chunking/anchoring.
_BLOCK_TAGS = frozenset(
    {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dt", "dd", "figcaption"}
)


def _local(tag: str) -> str:
    """Strip a namespace from a tag/attribute name."""
    return tag.rsplit("}", 1)[-1]


def sanitize(data: bytes) -> str:
    """Return the sanitized XHTML body fragment (or '' for an empty body).

    Raises :class:`xml.etree.ElementTree.ParseError` for unparseable input;
    callers decide how to classify that.
    """
    ET.register_namespace("", XHTML_NS)
    root = ET.fromstring(data)
    _strip_elements(root)
    _strip_attrs(root)
    body = root.find(f"{{{XHTML_NS}}}body")
    if body is None:
        body = root
    return ET.tostring(body, encoding="unicode")


def _local_name(el: ET.Element) -> str | None:
    """The local (namespace-stripped) tag name, or None for nameless nodes.

    Comments and processing instructions carry a *callable* in ``tag`` at
    runtime; typeshed types the tree as named elements only, so the tag is
    widened through ``object`` to keep the runtime guard type-checkable.
    """
    tag: object = el.tag
    if not isinstance(tag, str):
        return None
    return _local(tag)


def _strip_elements(el: ET.Element) -> None:
    """Recursively remove forbidden elements (and comments/PIs)."""
    for child in list(el):
        name = _local_name(child)
        if name is None or name.lower() in _FORBIDDEN_TAGS:
            el.remove(child)
            continue
        _strip_elements(child)


def _strip_attrs(el: ET.Element) -> None:
    """Drop event handlers and active/external URIs, keep everything else."""
    for child in el.iter():
        if _local_name(child) is None:
            continue
        for name in list(child.attrib):
            lname = _local(name).lower()
            if lname.startswith("on"):
                del child.attrib[name]
                continue
            if lname in _URI_ATTRS:
                value = child.attrib[name]
                if _EXTERNAL_URI.match(value) or value.lstrip().lower().startswith("//"):
                    del child.attrib[name]


def parse_blocks(sanitized: str) -> tuple[list[str], str | None]:
    """Return ``(paragraphs, title)`` from a sanitized XHTML fragment.

    *paragraphs* is text in document order (block-level elements only);
    *title* is the first non-empty ``<h1>``/``<h2>`` text, if any.
    """
    if not sanitized:
        return [], None
    root = ET.fromstring(sanitized)
    paragraphs: list[str] = []
    title: str | None = None
    for el in root.iter():
        tag = _local_name(el)
        if tag is None:
            continue
        tag = tag.lower()
        if tag in _BLOCK_TAGS:
            text = "".join(el.itertext()).strip()
            if text:
                paragraphs.append(text)
        if title is None and tag in ("h1", "h2"):
            text = "".join(el.itertext()).strip()
            if text:
                title = text
    return paragraphs, title
