"""Reconciliation primitives (PRD §7, §8A, M1).

The recovery principle: **a missing mount must never cause content to be
deleted.** When a source mount is absent we cannot tell a "deleted book" from a
"disconnected disk", so we treat the whole mount as *unavailable* and touch
nothing on it. Only for a mount verified present (via its sentinel) do we prune
the *alias rows* pointing at files that are gone — and even then we prune only
the alias, never the document, the revision, or the archived original.

Content garbage collection (documents/revisions/archived objects with no live
reference) is deliberately *not* done here; it is an explicit, reference-checked
operation in a later milestone (M7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .db import Database
from .identity import normalize_path

__all__ = ["Mount", "ReconcileReport", "mount_present", "reconcile_catalog"]


@dataclass(frozen=True)
class Mount:
    """A configured source root and the sentinel proving the mount is present.

    *sentinel* is ``None`` when no sentinel is configured, in which case the
    mount is treated as present (the root directory existing is taken as the
    signal) rather than as "missing but unprovable."
    """

    label: str
    root: Path
    sentinel: Path | None = None


@dataclass
class ReconcileReport:
    unavailable_mounts: list[str] = field(default_factory=list)
    pruned_aliases: int = 0
    deleted_content: int = 0  # always 0 in M1: no content GC here
    messages: list[str] = field(default_factory=list)


def mount_present(mount: Mount) -> bool:
    if mount.sentinel is not None:
        return mount.sentinel.exists()
    return mount.root.exists()


def reconcile_catalog(
    db: Database,
    mounts: list[Mount],
    visible_paths: set[str],
) -> ReconcileReport:
    """Reconcile the catalog against what is actually visible on each mount.

    * *visible_paths* are the normalized source paths currently present.
    * A mount whose sentinel is absent is reported as unavailable and skipped —
      nothing on it is pruned, so an absent disk can never delete content.
    * For present mounts, alias rows whose path is no longer visible are removed.
      Documents, revisions, and archived originals are left untouched.
    """
    report = ReconcileReport()
    for mount in mounts:
        if not mount_present(mount):
            report.unavailable_mounts.append(mount.label)
            report.messages.append(f"mount {mount.label!r} unavailable; leaving content intact")
            continue
        root_norm = normalize_path(mount.root)
        rows = db.query(
            "SELECT path FROM path_aliases WHERE path LIKE ? ORDER BY path",
            (root_norm + "/%",),
        )
        stale = [r["path"] for r in rows if r["path"] not in visible_paths]
        if not stale:
            continue
        with db.transaction():
            cur = db.execute(
                f"DELETE FROM path_aliases WHERE path IN ({','.join('?' * len(stale))})",
                stale,
            )
            report.pruned_aliases = cur.rowcount
    return report
