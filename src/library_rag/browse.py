"""Filesystem browser for the web UI (M10).

A purely read-only view over the configured book tree
(``cfg.browse.root``): list one level of a directory relative to the root,
filter files by the configured ``file_types``, and attach the *active*
revision id of each file the catalog knows about.

The browser never leaves the root. The request path must be relative; the
fully-joined path is containment-checked on its ``os.path.realpath``
against the root's realpath, which kills ``..`` escapes, absolute paths, and
symlinks that point out of the tree — matching the project's
``scan.follow_symlinks: False`` stance. Symlinked entries are skipped in
the listing entirely (neither a directory to drill into nor a file to
open), again matching ``iter_candidate_paths``.

This module is pure (no FastAPI): ``resolve_browse_path`` /
``list_browse_dir`` / ``lookup_active_revisions`` are unit-testable without
a running app. The stored résumé is deliberately NOT served here — the
client reuses ``GET /resumes/{rev}`` for right-click.
"""

from __future__ import annotations

import contextlib
import os
import posixpath
from pathlib import Path
from typing import Any

from .config import _is_within
from .db import Database
from .identity import normalize_path

__all__ = [
    "BrowseError",
    "BrowseNotFound",
    "list_browse_dir",
    "lookup_active_revisions",
    "resolve_browse_path",
]


class BrowseError(Exception):
    """A browse request that is malformed or escapes the configured root.

    The API layer maps this to 400.
    """


class BrowseNotFound(BrowseError):
    """The path is inside the root but is not an existing directory.

    The API layer maps this to 404.
    """


def resolve_browse_path(root: os.PathLike[str] | str, rel: str) -> str:
    """Resolve *rel* (relative to *root*) to a safe absolute path.

    Returns the normalized **non-resolved** absolute POSIX path (the same
    normalization the catalog stores, so stored paths line up for lookup).
    Raises :class:`BrowseError` when *rel* is absolute, contains a null
    byte, or the resolved path would leave the root — including through a
    symlink inside the tree.

    The containment check compares realpaths (both sides resolved) so a
    root that is itself a symlink still works; the returned path is the
    un-resolved form on purpose — the catalog indexes un-resolved paths,
    and scan and browse share the same configured root string.
    """
    if not isinstance(rel, str) or "\x00" in rel:
        raise BrowseError("invalid browse path")
    rel = rel.strip()
    if rel.endswith("/"):
        rel = rel.rstrip("/")
    if not rel or rel == ".":
        rel = ""
    if os.path.isabs(rel):
        raise BrowseError("browse path must be relative to the root")
    base = os.path.abspath(str(root))
    joined = os.path.normpath(os.path.join(base, rel)) if rel else base
    # Containment on the realpaths (resolves every symlink on both sides).
    if not _is_within(Path(os.path.realpath(joined)), Path(os.path.realpath(base))):
        raise BrowseError("browse path escapes the root")
    return normalize_path(joined)


def lookup_active_revisions(
    db: Database, abs_paths: list[str]
) -> dict[str, dict[str, Any]]:
    """Map absolute (normalized) file paths to their doc's active revision.

    ``path_aliases`` holds every registered path (anchor, rename, and
    duplicate rows). The join is on the *active* revision, not the alias's
    own rev — a stale duplicate alias can point at a now-inactive revision,
    and the browser must always show the document's current one (the same
    rule ``/library`` applies).
    """
    if not abs_paths:
        return {}
    marks = ",".join("?" * len(abs_paths))
    rows = db.query(
        f"""
        SELECT a.path AS path, r.rev_id AS rev_id, r.size_bytes AS size_bytes
        FROM path_aliases a
        JOIN source_revisions r ON r.doc_id = a.doc_id AND r.is_active = 1
        WHERE a.path IN ({marks})
        """,
        tuple(abs_paths),
    )
    return {
        str(row["path"]): {
            "rev_id": row["rev_id"],
            "size_bytes": int(row["size_bytes"]),
        }
        for row in rows
    }


def list_browse_dir(
    db: Database,
    root: os.PathLike[str] | str,
    rel: str,
    file_types: list[str],
) -> dict[str, object]:
    """List one level of *rel* under *root* (directories first, then files).

    Directories are always listed (drill-down); files are listed only when
    their name ends with one of *file_types* (already lowercase-
    normalized by the config validator; compared case-insensitively).
    Each file row carries the catalog's active ``rev_id`` + title + size
    when the path is indexed, else ``None`` fields ("not in the index
    yet"). Symlinked entries are skipped (see module docstring).

    Returns ``{"path": <normalized rel>, "entries": [...]}``.
    Raises :class:`BrowseError` for a malformed/escaping path and
    :class:`BrowseNotFound` when the resolved path is not an existing
    directory (the route maps those to 400 and 404 respectively).
    """
    abs_dir = resolve_browse_path(root, rel)
    if not os.path.isdir(abs_dir):
        raise BrowseNotFound("not a directory")
    suffixes = tuple(ft for ft in file_types if ft)
    norm_rel = posixpath.normpath(rel.strip().rstrip("/")) if rel.strip() else ""
    if norm_rel in (".", ""):
        norm_rel = ""
    dirs: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    with os.scandir(abs_dir) as it:
        for entry in it:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if entry.is_symlink():
                continue  # scan follows no symlinks; browse lists none either
            if not is_dir and not entry.name.lower().endswith(suffixes):
                continue
            child_rel = f"{norm_rel}/{entry.name}" if norm_rel else entry.name
            row: dict[str, object] = {
                "name": entry.name,
                "path": child_rel,
                "is_dir": is_dir,
                "size_bytes": None,
                "rev_id": None,
                "title": None,
            }
            if is_dir:
                dirs.append(row)
            else:
                with contextlib.suppress(OSError):
                    row["size_bytes"] = entry.stat(follow_symlinks=False).st_size
                files.append(row)
    # Batched catalog lookup for the listed files (one IN (...) query).
    abs_names = {
        str(row["name"]): normalize_path(os.path.join(abs_dir, str(row["name"])))
        for row in files
    }
    indexed = lookup_active_revisions(db, list(abs_names.values()))
    for row in files:
        hit = indexed.get(abs_names[str(row["name"])])
        if hit:
            row["rev_id"] = hit["rev_id"]
            row["title"] = os.path.splitext(str(row["name"]))[0]
            row["size_bytes"] = int(hit["size_bytes"])
    dirs.sort(key=lambda r: str(r["name"]).lower())
    files.sort(key=lambda r: str(r["name"]).lower())
    return {"path": norm_rel, "entries": dirs + files}
