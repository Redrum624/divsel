"""Coverage for the CI-only Python and YAML under ``.github/``.

``assemble_matrix.py`` and the workflow files had no local signal at all: a
renamed key, a bad ``runs-on`` or an exception in the table builder was first
visible as a red -- or vacuously green -- CI run after a push. The workflow
checks here are a lint, not a schema validation: they assert the invariants this
repository actually depends on.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(path: Path):
    if not path.exists():  # an installed copy of this suite has no .github/
        pytest.skip(f"{path} is not present")
    spec = importlib.util.spec_from_file_location(f"_ci_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(library: str, os_name: str, python: str, status: str = "ok") -> dict:
    return {
        "library": library,
        "os": os_name,
        "python": python,
        "status": status,
        "first_line": "boom | with a pipe" if status != "ok" else "",
        "tail": ["last line"],
    }


def _run_assemble(monkeypatch, cells_dir: Path, out: Path) -> tuple[int, str]:
    assemble = _load(SCRIPTS / "assemble_matrix.py")
    monkeypatch.setattr(sys, "argv", ["assemble_matrix.py", str(cells_dir), str(out)])
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = assemble.main()
    return code, buffer.getvalue()


def test_assemble_matrix_with_no_cells_says_so_and_fails(monkeypatch, tmp_path):
    """The zero-record empty state.

    Every ``cell`` step is ``continue-on-error`` and ``assemble`` is
    ``if: always()``, so twelve failed cells used to produce a one-column table
    with four empty rows, ``0 cells``, and a green job. A failed install is a
    measurement; no measurement at all is a broken run.
    """
    cells = tmp_path / "cells"
    cells.mkdir()
    out = tmp_path / "install-matrix.json"

    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 1
    assert "No cells" in text
    assert "| library |" not in text, "a degenerate table was printed anyway"
    assert not out.exists()


def test_assemble_matrix_builds_the_table_from_the_records(monkeypatch, tmp_path):
    cells = tmp_path / "cells"
    (cells / "linux").mkdir(parents=True)
    records = [
        _record("divsel", "Linux", "3.12"),
        _record("divsel", "Windows", "3.11"),
        _record("gist-select", "Linux", "3.12", status="fail"),
    ]
    for i, rec in enumerate(records):
        (cells / "linux" / f"{i}.json").write_text(json.dumps(rec), encoding="utf-8")
    out = tmp_path / "install-matrix.json"

    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    # OS_ORDER first, then anything unexpected, and the pythons numerically.
    assert written["oses"] == ["Linux", "Windows"]
    assert written["pythons"] == ["3.11", "3.12"]
    assert len(written["cells"]) == 3

    lines = text.splitlines()
    header = next(ln for ln in lines if ln.startswith("| library |"))
    assert header.count("|") == 2 + len(written["oses"]) * len(written["pythons"])
    # A library with no record at all is "not run", not a crash.
    submodlib = next(ln for ln in lines if ln.startswith("| submodlib-py |"))
    assert submodlib.count("not run") == 4
    # A pipe inside pip's output would otherwise split the row.
    failed = next(ln for ln in lines if ln.startswith("| gist-select |"))
    assert r"boom \| with a pipe" in failed
    assert "3 cells" in text


# --------------------------------------------------------------------------- #
# .github/workflows/*.yml                                                     #
# --------------------------------------------------------------------------- #


def _workflows():
    yaml = pytest.importorskip("yaml", reason="PyYAML is needed to lint the workflows")
    if not WORKFLOWS.is_dir():
        pytest.skip(f"{WORKFLOWS} is not present")
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, "no workflow files found"
    return [(path, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in files]


def test_every_workflow_parses_and_declares_jobs():
    for path, doc in _workflows():
        assert isinstance(doc, dict), path.name
        # `on:` is YAML 1.1's boolean True once parsed -- both spellings count.
        assert True in doc or "on" in doc, f"{path.name} has no trigger"
        assert doc.get("jobs"), f"{path.name} declares no jobs"


# The GitHub-hosted runner labels this repository is allowed to name. A label
# outside this set is either a typo or a runner image nobody here has checked
# exists -- both cost a push to discover, which is what this file is for.
KNOWN_RUNNER_LABELS = {
    "ubuntu-latest",
    "ubuntu-24.04",
    "ubuntu-22.04",
    "ubuntu-24.04-arm",
    "ubuntu-22.04-arm",
    "windows-latest",
    "windows-2025",
    "windows-2022",
    "windows-11-arm",
    "macos-latest",
    "macos-15",
    "macos-14",
    "macos-13",
}


def test_every_job_names_a_runner_that_exists():
    """The typo scenario: ``runs-on: ${{ matrix.oss }}`` is silently empty, and
    ``runs-on: ubuntu-24.04-armm`` is a label GitHub does not serve.

    A misspelled matrix key in ``runs-on`` is not a YAML error; it expands to
    nothing and the job fails at dispatch. A misspelled *label* is not an error
    either -- the job simply never gets a runner. Both are only visible after a
    push, so both are checked here: the matrix key must be declared by the job,
    and every literal label -- hard-coded or listed in the matrix it comes from
    -- must be one of ``KNOWN_RUNNER_LABELS``.
    """
    expr = re.compile(r"\$\{\{\s*matrix\.([A-Za-z_][A-Za-z0-9_-]*)\s*\}\}")
    for path, doc in _workflows():
        for name, job in doc["jobs"].items():
            where = f"{path.name}:{name}"
            if "uses" in job:  # a reusable-workflow call has no runs-on
                continue
            runs_on = job.get("runs-on")
            assert runs_on, f"{where} has no runs-on"
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            declared = set(matrix)
            keys = expr.findall(str(runs_on))
            for key in keys:
                assert key in declared, (
                    f"{where}: runs-on uses matrix.{key}, which the job does not declare "
                    f"(declared: {sorted(declared)})"
                )
                # The values that key expands to are labels too. `include` rows
                # can add more; those are checked as well when they name it.
                values = matrix[key]
                if isinstance(values, list):
                    for value in values:
                        assert value in KNOWN_RUNNER_LABELS, (
                            f"{where}: matrix.{key} offers {value!r}, which is not a "
                            f"runner label this repository has checked"
                        )
                for row in matrix.get("include") or []:
                    if key in row:
                        assert row[key] in KNOWN_RUNNER_LABELS, (
                            f"{where}: matrix include sets {key}={row[key]!r}, which is "
                            f"not a runner label this repository has checked"
                        )
            if not keys and "${{" not in str(runs_on):
                assert runs_on in KNOWN_RUNNER_LABELS, (
                    f"{where}: runs-on {runs_on!r} is not a runner label this "
                    f"repository has checked (typo, or an image nobody verified)"
                )
            assert job.get("steps"), f"{where} has no steps"
            for i, step in enumerate(job["steps"]):
                assert "uses" in step or "run" in step, f"{where} step {i} does neither"


# Every command that runs one of the two golden readers. BOTH readers skip a
# fixture they cannot find -- right for a published `.crate` or an installed
# copy of the suite, wrong for the job whose whole purpose is that fixture --
# and both honour DIVSEL_REQUIRE_GOLDEN. Listing only the Rust spellings here
# is how the Python half of the same contract went ungated.
GOLDEN_READER_COMMANDS = (
    "--test golden",
    "cargo test --workspace",
    "pytest python/tests -q",
    "pytest python/tests/test_golden.py",
)


def test_the_golden_gate_cannot_skip_in_ci():
    """``DIVSEL_REQUIRE_GOLDEN`` is what stops either reader skipping in CI."""
    checked = 0
    for path, doc in _workflows():
        if path.name != "ci.yml":
            continue
        for name, job in doc["jobs"].items():
            for step in job["steps"]:
                run = str(step.get("run", ""))
                if any(command in run for command in GOLDEN_READER_COMMANDS):
                    env = {**(doc.get("env") or {}), **(job.get("env") or {}), **(step.get("env") or {})}
                    assert env.get("DIVSEL_REQUIRE_GOLDEN") not in (None, "", "0"), (
                        f"ci.yml:{name} runs a golden reader without DIVSEL_REQUIRE_GOLDEN"
                    )
                    checked += 1
        break
    else:  # pragma: no cover - ci.yml is in the repository
        pytest.fail("ci.yml not found")
    # Both readers, or this lint is watching a workflow that moved.
    assert checked >= 4, f"only {checked} golden-reader steps found in ci.yml"


def test_assemble_matrix_names_a_malformed_record(monkeypatch, tmp_path):
    """A cell record missing a key the table is built on.

    install_cell.sh writes these and assemble_matrix.py is their only reader, so
    a key renamed on one side used to surface as a bare ``KeyError`` from inside
    a comprehension -- naming neither the key nor the file it came from.
    """
    cells = tmp_path / "cells"
    cells.mkdir()
    good = _record("divsel", "Linux", "3.12")
    (cells / "a.json").write_text(json.dumps(good), encoding="utf-8")
    broken = {k: v for k, v in good.items() if k != "python"}
    (cells / "b.json").write_text(json.dumps(broken), encoding="utf-8")
    out = tmp_path / "install-matrix.json"

    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 1
    assert "Malformed cell record" in text
    assert "b.json" in text and "`python`" in text
    assert "| library |" not in text
    assert not out.exists()


def test_install_cell_writes_every_key_assemble_matrix_reads():
    """The two halves of the cell-record contract, pinned to each other.

    ``install_cell.sh`` is a bash script with no test of any kind, and its
    record-writing heredoc is the only producer of the keys the matrix builder
    indexes. A rename on either side is invisible until a workflow run.
    """
    script = SCRIPTS / "install_cell.sh"
    if not script.exists():  # pragma: no cover - an installed copy has no .github/
        pytest.skip(f"{script} is not present")
    written = set(re.findall(r'"([a-z_]+)":', script.read_text(encoding="utf-8")))
    assemble = (SCRIPTS / "assemble_matrix.py").read_text(encoding="utf-8")
    read = set(re.findall(r'r\[?\.?get\("([a-z_]+)"\)|r\["([a-z_]+)"\]', assemble))
    needed = {name for pair in read for name in pair if name}
    needed |= {"first_line", "tail"}  # read through `r.get(...)`/indexing above
    missing = needed - written
    assert not missing, f"install_cell.sh writes no {sorted(missing)}"
