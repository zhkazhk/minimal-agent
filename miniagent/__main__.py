"""让 `python -m miniagent ...` 直接可用。"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
