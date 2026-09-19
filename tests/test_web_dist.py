"""M5 research UI bundle guard (PRD §10): the committed ``web/dist`` must be
present for the self-hosted gate and must load nothing from the network —
no CDN scripts, no absolute external assets, and the pdf.js worker shipped
as a same-origin asset (the server CSP allows only ``'self'`` workers).
"""

from __future__ import annotations

import re
from pathlib import Path

from library_rag.api import find_web_dist

DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


def test_web_dist_present_and_discoverable() -> None:
    d = find_web_dist()
    assert d is not None, "web/dist must be built and committed (self-hosted UI)"
    assert (d / "index.html").is_file()


def test_index_html_only_local_relative_assets() -> None:
    html = (DIST / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs, "index.html must reference the built JS/CSS assets"
    for ref in refs:
        assert not ref.startswith(("http://", "https://", "//", "/")), (
            f"non-local asset ref: {ref}"
        )
        assert (DIST / ref.removeprefix("./")).is_file(), (
            f"referenced asset missing from dist: {ref}"
        )


def test_no_external_resource_loads_in_bundle() -> None:
    files = sorted(DIST.glob("assets/*"))
    assert files, "dist/assets must contain the built bundle"
    patterns = (
        r'src=["\']https?://',  # injected <script src="http...">
        r'new Worker\(["\']https?://',  # cross-origin worker
        r'import\(["\']https?://',  # dynamic import from a URL
        r'url\(\s*["\']?https?://',  # CSS url(http...)
    )
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for pat in patterns:
            m = re.search(pat, text)
            assert m is None, f"{f.name}: external load pattern {pat!r} matched {m.group(0)!r}"


def test_pdfjs_worker_bundled_locally() -> None:
    workers = list(DIST.glob("assets/pdf.worker*.mjs"))
    assert workers, "pdf.js worker must ship as a same-origin asset"
    js_files = [f for f in DIST.glob("assets/*.js") if "pdf.worker" not in f.name]
    assert js_files
    for f in js_files:
        assert any(
            w.name in f.read_text(encoding="utf-8", errors="replace") for w in workers
        ), f"{f.name} does not reference the bundled worker"
