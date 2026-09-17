"""library_rag — local-first RAG research application for a personal book library.

Package layout (grows across milestones; see CLAUDE_PRD.md §4):

    config    -- config loading/validation (M0)
    log       -- structured logging (M0)
    doctor    -- read-only environment diagnostics (M0)
    cli       -- command-line interface (M0 skeleton, grows)
    db        -- SQLite schema + migrations (M1)
    catalog   -- documents / revisions / aliases (M1)
    jobs      -- durable job leases + fencing (M1)
    artifacts -- atomic artifact commit protocol (M1)
    extraction -- PDF / OCR / EPUB (M2/M3)
    chunking  -- token-aware chunking (M3)
    embeddings -- checkpointed embedding workers (M4)
    indexing  -- Qdrant publication/reconciliation (M4)
    retrieval -- fusion + reranking (M4)
    citations -- evidence manifests + citation validation (M5)
    api       -- FastAPI app (M5)
"""

__version__ = "0.0.1"
