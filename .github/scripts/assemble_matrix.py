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


def _version_key(version: str) -> tuple[int, ...]:
    """``"3.12"`` -> ``(3, 12)``; anything unparseable -> ``()``.

    Both the guard and the sort use this, so "the value the guard accepted" and
    "the value the sort parses" cannot be two different judgements.
    """
    parts = version.split(".")
    if not all(part.isdigit() for part in parts) or not parts:
        return ()
    return tuple(int(part) for part in parts)


def main() -> int:
    cells_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cells_dir.rglob("*.json"))]
    if not records:
        # Every `cell` step is `continue-on-error` and this job is `if: always()`,
        # so "no artifacts at all" used to print a one-column table with four
        # empty rows and `0 cells`, green. A failed install is a measurement; no
        # measurement is a broken run, and it says so and exits non-zero.
        print("## Installability matrix")
        print("")
        print(f"**No cells**: no JSON records under `{cells_dir}`.")
        print("")
        print("The measurement did not happen -- every `cell` job failed before")
        print("uploading, or the artifact download matched nothing. There is no")
        print("table to print and no matrix to write.")
        return 1
    # A record missing one of the keys the table is keyed on used to surface as
    # a bare `KeyError` from inside a comprehension, naming neither the key nor
    # the file. install_cell.sh writes these, and nothing but this script reads
    # them, so a renamed key there is exactly the failure that has to be
    # readable here. `tail` is in the list because line 82 indexes it: leaving it
    # out reproduced the same bare KeyError -- rc=1 with zero bytes on stdout, so
    # `tee -a $GITHUB_STEP_SUMMARY` wrote an empty summary.
    required = ("library", "os", "python", "status", "tail")
    malformed = [
        (path, f"has no `{key}`")
        for path, record in zip(sorted(cells_dir.rglob("*.json")), records)
        for key in required
        if key not in record
    ]
    # Presence is not validity: `python` is parsed into a sort key, and
    # install_cell.sh can write an empty one -- `PY_VER="$(python -c ...)"` is ""
    # when that command fails, and `set -u` without `set -e` lets the script
    # carry on and write the record anyway. Unchecked, that was a `ValueError`
    # out of the sort lambda with the same empty-summary outcome.
    malformed += [
        (path, f"has an unusable `python` value {record['python']!r}")
        for path, record in zip(sorted(cells_dir.rglob("*.json")), records)
        if "python" in record and not _version_key(record["python"])
    ]
    if malformed:
        print("## Installability matrix")
        print("")
        for path, problem in malformed:
            print(f"**Malformed cell record**: `{path}` {problem}.")
        print("")
        print("install_cell.sh writes these records and this script is their only")
        print("reader; a key renamed on one side is a broken run, not a blank cell.")
        return 1
    pythons = sorted({r["python"] for r in records}, key=_version_key)
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
