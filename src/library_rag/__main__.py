"""Allow ``python -m library_rag``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
