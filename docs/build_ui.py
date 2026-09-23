"""Rebuild the demo pages from a fresh trace.

    python examples/assistant_demo.py --dir /tmp/p --trace docs/assistant-trace.json
    python docs/build_ui.py

Each page carries its trace inline (one `const TRACE = {...};` line) so it is a single file that
works offline and can be published anywhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = {"assistant-ui.html": "assistant-trace.json", "demo-ui.html": "demo-trace.json"}


def main() -> int:
    for page_name, trace_name in PAGES.items():
        page, trace = HERE / page_name, HERE / trace_name
        if not (page.exists() and trace.exists()):
            print(f"skipped {page_name}: missing page or trace")
            continue
        data = json.dumps(json.loads(trace.read_text(encoding="utf-8")), separators=(",", ":"))
        text, count = re.subn(r"const TRACE = .*?;\n", f"const TRACE = {data};\n",
                              page.read_text(encoding="utf-8"), count=1, flags=re.S)
        if not count:
            print(f"skipped {page_name}: no `const TRACE = ...;` line to replace")
            continue
        page.write_text(text, encoding="utf-8")
        print(f"{page_name}: {round(len(text.encode()) / 1024)} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
