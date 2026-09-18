"""Selective-OCR routing (PRD §8C).

Each extracted PDF page is assigned exactly one route:

* ``reuse`` — the native text layer is good; no OCR.
* ``ocr``   — the page needs Tesseract: no text at all (scanned page), or a
  broken text layer (``many_replacement``), or a sparse layer where embedded
  images cover enough of the page that the sparse text is almost certainly a
  partial layer over an image.
* ``skip``  — no text and no significant image coverage either: a legitimate
  blank or illustration page; OCR would only add noise.

Routing is a pure function of the extraction quality flags and the page's
image-area ratio, so it is reproducible and unit-testable without a Tesseract
binary. The image area comes from ``page.get_image_info()`` bboxes over the
rotation-aware page rect.
"""

from __future__ import annotations

from typing import Any

from ..config import OcrSettings

__all__ = ["ROUTE_OCR", "ROUTE_REUSE", "ROUTE_SKIP", "image_area_ratio", "route_page"]

ROUTE_REUSE = "reuse"
ROUTE_OCR = "ocr"
ROUTE_SKIP = "skip"


def image_area_ratio(page: Any) -> float:
    """Share of the page area covered by embedded images, clamped to [0, 1].

    Overlapping images are counted once (union via per-pixel coverage would be
    far more expensive than the error this approximation adds at threshold
    comparisons).
    """
    page_area = float(page.rect.width) * float(page.rect.height)
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        bbox = info.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        covered += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return min(1.0, covered / page_area)


def route_page(flags: list[str], image_ratio: float, settings: OcrSettings) -> str:
    """Decide the OCR route for one page from its flags and image coverage."""
    if "many_replacement" in flags:
        return ROUTE_OCR
    if "no_text" in flags or "sparse" in flags:
        # A scanned page's image must plausibly be the source of the missing
        # text; below the threshold the page is a legit blank/illustration.
        if image_ratio >= settings.image_area_threshold:
            return ROUTE_OCR
        return ROUTE_SKIP
    return ROUTE_REUSE
