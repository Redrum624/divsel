#!/usr/bin/env python
"""Merge the per-cell JSON records written by install_cell.sh into one matrix file and print a
Markdown table (the workflow appends it to $GITHUB_STEP_SUMMARY).

    python .github/scripts/assemble_matrix.py <cells dir> <out json>
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

LIBRARIES = ["gist-select", "submodlib-py", "gist-sampling", "divsel"]
OS_ORDER = ["Linux", "Windows", "macOS"]


def main() -> int:
    cells_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cells_dir.rglob("*.json"))]
    pythons = sorted({r["python"] for r in records}, key=lambda v: tuple(int(x) for x in v.split(".")))
    oses = [o for o in OS_ORDER if any(r["os"] == o for r in records)]
    oses += sorted({r["os"] for r in records} - set(oses))
    by_key = {(r["library"], r["os"], r["python"]): r for r in records}
    matrix = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "libraries": LIBRARIES,
        "oses": oses,
        "pythons": pythons,
        "cells": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=1), encoding="utf-8")

    lines = ["## Installability matrix", "", "| library | " + " | ".join(f"{o} / py{p}" for o in oses for p in pythons) + " |",
             "|---|" + "---|" * (len(oses) * len(pythons))]
    for lib in LIBRARIES:
        row = [lib]
        for o in oses:
            for p in pythons:
                r = by_key.get((lib, o, p))
                if r is None:
                    row.append("not run")
                elif r["status"] == "ok":
                    row.append("ok")
                else:
                    first = r.get("first_line") or (r["tail"][-1] if r["tail"] else "")
                    row.append("fail: `" + first.replace("|", "\\|")[:90] + "`")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", f"{len(records)} cells; JSON at `{out}`."]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
