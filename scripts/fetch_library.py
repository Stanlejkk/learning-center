"""Download the offline library PDFs listed in library/RESOURCES.md into library/pdfs/.

The PDFs are not stored in the git repo (they belong to their authors / publishers);
this script fetches them from their original public sources for your personal use.

    uv run python scripts/fetch_library.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESOURCES = ROOT / "library" / "RESOURCES.md"
OUT = ROOT / "library" / "pdfs"
ROW = re.compile(r"^\|\s*`([\w.-]+\.pdf)`\s*\|\s*(https?://\S+)\s*\|")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [m.groups() for line in RESOURCES.read_text(encoding="utf-8").splitlines() if (m := ROW.match(line))]
    failed = 0
    for name, url in rows:
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  have  {name}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "learning-center-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(data)
            print(f"  got   {name} ({len(data) / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}  ({url})")
    print(f"{len(rows) - failed}/{len(rows)} PDFs in {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
