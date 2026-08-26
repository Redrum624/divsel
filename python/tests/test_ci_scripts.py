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
import os
import re
import shutil
import subprocess
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
    # Intel macOS is its own label since macOS 15; release.yml's x86_64 wheel
    # job names it. Checked against
    # https://docs.github.com/en/actions/reference/runners/github-hosted-runners
    # on 2026-08-25.
    "macos-15-intel",
    "macos-14",
    "macos-13",
}


def _runner_labels(runs_on, matrix):
    r"""Every literal label ``runs-on`` can expand to, or ``None`` if unreadable.

    ``runs-on`` is either a literal label or an expression naming a matrix key --
    possibly a **dotted** one, ``${{ matrix.platform.runner }}``, which is what
    ``maturin generate-ci`` writes and what the previous regex
    (``matrix\.([A-Za-z_][A-Za-z0-9_-]*)\s*\}\}``) could not match. It found no
    keys, and the literal branch was then skipped because ``"${{" in runs_on``,
    so release.yml's ``linux``, ``windows`` and ``macos`` jobs -- the entire
    wheel build and publish, the one workflow whose failure loses the release --
    got no runner check of any kind.
    """
    text = str(runs_on)
    paths = re.findall(r"\$\{\{\s*matrix\.([A-Za-z_][A-Za-z0-9_.-]*?)\s*\}\}", text)
    if not paths:
        return None if "${{" in text else [text]
    labels = []
    for path in paths:
        key, *rest = path.split(".")
        assert key in matrix, (
            f"runs-on uses matrix.{path}, whose key {key!r} the job does not "
            f"declare (declared: {sorted(matrix)})"
        )
        declared = matrix[key]
        rows = list(declared) if isinstance(declared, list) else [declared]
        # `include` rows can add more values for the same key.
        rows += [row[key] for row in (matrix.get("include") or []) if key in row]
        for row in rows:
            value = row
            for step in rest:
                value = value.get(step) if isinstance(value, dict) else None
            if value is not None:
                labels.append(value)
    return labels


def test_every_job_names_a_runner_that_exists():
    """The typo scenario: ``runs-on: ${{ matrix.oss }}`` is silently empty, and
    ``runs-on: ubuntu-24.04-armm`` is a label GitHub does not serve.

    A misspelled matrix key in ``runs-on`` is not a YAML error; it expands to
    nothing and the job fails at dispatch. A misspelled *label* is not an error
    either -- the job simply never gets a runner. Both are only visible after a
    push, so both are checked here: the matrix key must be declared by the job,
    and every literal label -- hard-coded, listed in the matrix it comes from, or
    reached through a **dotted** path like ``matrix.platform.runner`` -- must be
    one of ``KNOWN_RUNNER_LABELS``. Every job must yield at least one such label,
    which is what stops a ``runs-on`` this lint cannot read from passing green.
    """
    for path, doc in _workflows():
        for name, job in doc["jobs"].items():
            where = f"{path.name}:{name}"
            if "uses" in job:  # a reusable-workflow call has no runs-on
                continue
            runs_on = job.get("runs-on")
            assert runs_on, f"{where} has no runs-on"
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            labels = _runner_labels(runs_on, matrix)
            assert labels, (
                f"{where}: runs-on {runs_on!r} resolves to no label this lint can "
                f"check -- an unreadable runs-on is not a checked one"
            )
            for label in labels:
                assert label in KNOWN_RUNNER_LABELS, (
                    f"{where}: runs-on can expand to {label!r}, which is not a runner "
                    f"label this repository has checked (typo, or an image nobody verified)"
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


def test_assemble_matrix_names_a_missing_tail_instead_of_raising_keyerror(
    monkeypatch, tmp_path
):
    """The guard required four keys; the table indexes five.

    ``r["tail"]`` at line 82 was read unguarded, so exactly the failure the
    guard was written to eliminate -- "a bare ``KeyError`` from inside a
    comprehension, naming neither the key nor the file" -- still happened for
    ``tail``: rc=1 and *zero* bytes on stdout, so ``| tee -a
    "$GITHUB_STEP_SUMMARY"`` wrote an empty job summary with no named error, no
    table and no cell. ``test_install_cell_writes_every_key_assemble_matrix_reads``
    already puts ``tail`` in the contract, so the two halves disagreed.
    """
    cells = tmp_path / "cells"
    cells.mkdir()
    broken = _record("gist-select", "Linux", "3.12", status="fail")
    del broken["tail"]
    (cells / "a.json").write_text(json.dumps(broken), encoding="utf-8")
    out = tmp_path / "install-matrix.json"

    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 1
    assert "Malformed cell record" in text
    assert "a.json" in text and "`tail`" in text
    assert not out.exists()


def test_assemble_matrix_names_an_unusable_python_version(monkeypatch, tmp_path):
    """The guard checked key presence, never value validity.

    ``tuple(int(x) for x in v.split("."))`` crashes on an empty or non-numeric
    ``python``, which ``install_cell.sh`` can write: ``PY_VER="$(python -c ...)"``
    yields ``""`` when that command fails, and ``set -u`` without ``set -e``
    lets the script carry on and write the record. Measured before this:
    ``ValueError: invalid literal for int() with base 10: ''``, rc=1, empty
    stdout -- a red ``assemble`` job with no diagnosis in the summary.
    """
    out = tmp_path / "install-matrix.json"
    for i, version in enumerate(("", "3.x", "three.twelve", "3..12")):
        cells = tmp_path / f"cells{i}"
        cells.mkdir()
        record = _record("divsel", "Linux", version)
        (cells / "a.json").write_text(json.dumps(record), encoding="utf-8")

        code, text = _run_assemble(monkeypatch, cells, out)
        assert code == 1, version
        assert "Malformed cell record" in text, version
        assert "a.json" in text and repr(version) in text, version
        assert "| library |" not in text, version
        assert not out.exists(), version


def test_assemble_matrix_reports_a_failed_cell_with_no_first_line_from_its_tail(
    monkeypatch, tmp_path
):
    """Both sub-branches of ``r.get("first_line") or (r["tail"][-1] if ...)``.

    Every ``_record`` in this file gives a truthy ``first_line``, which
    short-circuits the whole expression -- so the fallback to the log tail, the
    one thing that puts *something* in a failed cell when pip printed nothing on
    its first line, had never run.
    """
    cells = tmp_path / "cells"
    cells.mkdir()
    from_tail = _record("gist-select", "Linux", "3.12", status="fail")
    from_tail["first_line"] = ""
    from_tail["tail"] = ["...", "ERROR: no matching distribution"]
    (cells / "a.json").write_text(json.dumps(from_tail), encoding="utf-8")
    nothing = _record("submodlib-py", "Linux", "3.12", status="fail")
    nothing["first_line"] = ""
    nothing["tail"] = []
    (cells / "b.json").write_text(json.dumps(nothing), encoding="utf-8")
    out = tmp_path / "install-matrix.json"

    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 0
    lines = text.splitlines()
    assert "ERROR: no matching distribution" in next(
        ln for ln in lines if ln.startswith("| gist-select |")
    )
    # An empty tail is an empty cell body, not a crash and not "not run".
    empty = next(ln for ln in lines if ln.startswith("| submodlib-py |"))
    assert "fail: ``" in empty and "not run" not in empty


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


def test_no_workflow_gate_captures_output_it_only_prints_on_success():
    r"""``out=$(cmd 2>&1)`` then ``echo "$out"`` under ``set -e`` is backwards.

    The shell exits at the assignment when ``cmd`` fails, so the ``echo`` runs
    only on success: the output is printed when nobody needs it and discarded
    when they do. Verified: ``set -euo pipefail; out=$(echo 'important
    diagnostics'; exit 3); echo "$out"`` exits 3 having printed nothing. Both
    of ci.yml's grep gates were written that way, so a ``cargo test --benches``
    that failed to compile, or a failing aarch64 ``metric::`` run, turned the
    step red with an empty log.

    A gate must stream its command's output (``| tee``) so a failure is
    readable, and ``pipefail`` still fails the step.
    """
    for path, doc in _workflows():
        for name, job in doc["jobs"].items():
            if "uses" in job:
                continue
            for i, step in enumerate(job.get("steps") or []):
                script = step.get("run") or ""
                if "set -e" not in script:
                    continue
                assert not re.search(r"^\s*[A-Za-z_]\w*=\$\(", script, re.M), (
                    f"{path.name}:{name} step {i} ({step.get('name')!r}) captures a "
                    f"command's output into a variable under `set -e`, which prints it "
                    f"only when the command succeeded"
                )


# --------------------------------------------------------------------------- #
# docs prose vs git state                                                     #
# --------------------------------------------------------------------------- #

# Claims about CI having never happened. Each is falsified the moment a remote
# tracking branch exists, and no test compared any of them to git -- which is
# how "no workflow in this repository has ever run" survived three review
# rounds while `.git/logs/refs/remotes/origin/feat/v0.1` recorded nine pushes
# of a tree containing `.github/workflows/ci.yml`, whose trigger is
# `push: branches: [main, "feat/**"]`.
_STALE_CI_CLAIMS = (
    "has not been pushed to CI",
    "has never been pushed",
    "no workflow in this repository has ever run",
    "the workflow itself has not run",
    "not run yet",
    "never been through CI",
)

_CI_DOCS = ("docs/RELEASE.md", "docs/benchmarks/README.md", "README.md")


def test_no_doc_claims_ci_never_ran_while_git_records_pushes():
    """Prose that says "this has never reached CI" must agree with the refs.

    A reader following ``docs/RELEASE.md`` step 1 on the strength of such a
    claim runs ``gh repo create ... --source . --push`` and gets "Name already
    exists on this account".
    """
    git_dir = ROOT / ".git"
    if not git_dir.is_dir():  # pragma: no cover - an exported or installed tree
        pytest.skip(f"{git_dir} is not present")
    remotes = git_dir / "refs" / "remotes"
    logs = git_dir / "logs" / "refs" / "remotes"
    pushed = sorted(
        {p.name for p in remotes.rglob("*") if p.is_file()}
        | {p.name for p in logs.rglob("*") if p.is_file()}
    )
    if not pushed:
        pytest.skip("no remote-tracking refs: nothing has been pushed from here")

    for name in _CI_DOCS:
        path = ROOT / name
        if not path.exists():  # pragma: no cover
            continue
        text = path.read_text(encoding="utf-8")
        for claim in _STALE_CI_CLAIMS:
            assert claim not in text, (
                f"{name} says {claim!r}, but this checkout has pushed remote-tracking "
                f"refs ({', '.join(pushed)}); the workflows live in the pushed tree"
            )


def test_release_uploads_and_the_publish_glob_agree():
    """``release.yml`` has no assertion beyond "it parses and declares jobs".

    The publish job downloads every artifact and then names ``wheels-*/*`` twice
    -- once for the attestation, once for ``uv publish``. Renaming an upload to
    anything outside that glob publishes fewer wheels than were built, on a tag
    push, with both steps green: ``uv publish`` uploads whatever the glob
    matched. So the two halves are pinned to each other here.
    """
    path = WORKFLOWS / "release.yml"
    if not path.exists():  # pragma: no cover - an installed copy has no .github/
        pytest.skip(f"{path} is not present")
    yaml = pytest.importorskip("yaml", reason="PyYAML is needed to lint the workflows")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    uploads = [
        step["with"]["name"]
        for job in doc["jobs"].values()
        for step in (job.get("steps") or [])
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert uploads, "no upload-artifact step: the publish job would have nothing to publish"
    prefix = "wheels-"
    for name in uploads:
        assert name.startswith(prefix), (
            f"release.yml uploads {name!r}, which the publish job's {prefix}*/* glob "
            f"does not match -- those wheels would be built and never published"
        )

    publish = doc["jobs"]["release"]
    globs = [
        value
        for step in publish["steps"]
        for value in (
            [step.get("run", "")]
            + [str(v) for v in (step.get("with") or {}).values()]
        )
        if f"{prefix}*/*" in str(value)
    ]
    assert len(globs) == 2, (
        "expected the attestation subject-path and the uv publish argument to be the "
        f"only two {prefix}*/* globs, found {globs}"
    )
    # Every job whose wheels the release needs must actually be a dependency.
    building = {
        name
        for name, job in doc["jobs"].items()
        for step in (job.get("steps") or [])
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    }
    assert building <= set(publish["needs"]), (
        f"{sorted(building - set(publish['needs']))} upload wheels but the release job "
        f"does not need them, so it can publish before they exist"
    )


def test_install_cell_actually_runs_and_writes_a_record_assemble_can_read(
    monkeypatch, tmp_path
):
    """``install_cell.sh`` had no executable coverage at all.

    The only test touching it regex-greps its source text, so its venv trap, its
    ``INSTALL_RC``/``IMPORT_RC`` capture and its record-writing heredoc had never
    been run -- on any platform, by anything. This drives it end to end against a
    package that cannot exist, with ``PIP_NO_INDEX`` so pip fails offline in
    milliseconds, and then feeds the record it wrote through the assembler that
    is its only reader.

    Costs one ``python -m venv`` (about 8 s), which is the script under test.
    """
    script = SCRIPTS / "install_cell.sh"
    if not script.exists():  # pragma: no cover - an installed copy has no .github/
        pytest.skip(f"{script} is not present")
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - no POSIX shell on this host
        pytest.skip("bash is not on PATH")

    work = tmp_path / "work"
    work.mkdir()
    cells = tmp_path / "cells"
    env = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "RUNNER_OS": "Linux",
    }
    proc = subprocess.run(
        [
            bash,
            str(script),
            "divsel",
            "divsel-no-such-package-xyz",
            "divsel_no_such_module",
            str(cells),
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # "Never exits non-zero for an install failure -- the failure IS the
    # measurement."
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The venv trap cleaned up after itself, in the directory it ran in.
    assert not list(work.glob("venv-*")), "the EXIT trap left a venv behind"

    written = list(cells.glob("*.json"))
    assert len(written) == 1, written
    assert written[0].name.startswith("Linux-py"), written[0].name
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assemble = _load(SCRIPTS / "assemble_matrix.py")
    for key in ("library", "os", "python", "status", "tail"):
        assert key in record, key
    assert assemble._version_key(record["python"]), record["python"]
    assert record["status"] == "fail" and record["install_rc"] != 0
    assert record["import_rc"] != 0
    assert record["tail"], "pip printed nothing into the log"
    assert (cells / "divsel.log").exists()

    # And the assembler reads it without a word of special handling.
    out = tmp_path / "install-matrix.json"
    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 0, text
    assert "| divsel |" in text and "1 cells" in text
