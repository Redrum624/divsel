import os
import shutil
import subprocess
import sys

import divsel


def test_import():
    assert divsel.__version__ == "0.1.0"


def _import_in_child(extra_path: str | None = None) -> subprocess.CompletedProcess:
    """Import ``divsel`` with ``divsel._divsel`` refused, in a child process.

    A meta path finder reproduces "the extension will not import" without
    touching the installation, and the child keeps this process's own import
    intact. ``extra_path`` is prepended to ``sys.path``, so a copy of the package
    with no compiled module next to it can be imported instead of the real one.
    """
    code = (
        "import sys\n"
        f"extra = {extra_path!r}\n"
        "if extra:\n"
        "    sys.path.insert(0, extra)\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'divsel._divsel':\n"
        "            raise ImportError('DLL load failed: a dependency is missing')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "try:\n"
        "    import divsel\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('divsel imported without its extension module')\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_a_present_but_unloadable_extension_is_not_reported_as_unbuilt():
    """The ``except ImportError`` arm must not assert a cause it did not check.

    Any ImportError raised while loading the extension -- a missing C runtime, an
    ABI-mismatched numpy, a broken transitive DLL -- used to come back as "the
    compiled extension module (_divsel) is not built" with `maturin develop` as
    the fix, which is a rebuild of something the user already has. The real cause
    was only visible in the chained ``__cause__``.

    This venv has the extension installed, so the arm must take the "present but
    unloadable" branch and repeat the underlying message.
    """
    installed = [
        name
        for name in os.listdir(os.path.dirname(divsel.__file__))
        if name.startswith("_divsel.") and name.endswith((".pyd", ".so", ".dylib"))
    ]
    assert installed, "this test needs the built extension next to divsel/__init__.py"

    proc = _import_in_child()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "could not load it" in proc.stdout
    assert "DLL load failed: a dependency is missing" in proc.stdout
    assert "is not built" not in proc.stdout
    assert installed[0] in proc.stdout


def test_a_genuinely_missing_extension_still_points_at_maturin_develop(tmp_path):
    """The other branch: a package directory with no compiled module in it.

    This is the only test that reaches it. Blocking ``divsel._divsel`` inside a
    venv where the extension IS installed takes the present-but-unloadable arm
    above -- whose message contains both "maturin develop" and "_divsel", so a
    test asserting only those two strings passed while the "is not built" arm
    went unexercised. Hence the copied package directory with nothing next to
    it, and the ``is not built`` assertion below.
    """
    package = tmp_path / "divsel"
    package.mkdir()
    shutil.copy(
        os.path.join(os.path.dirname(divsel.__file__), "__init__.py"),
        package / "__init__.py",
    )
    assert not [p for p in package.iterdir() if p.suffix in {".pyd", ".so", ".dylib"}]

    proc = _import_in_child(extra_path=str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "is not built" in proc.stdout
    assert "maturin develop" in proc.stdout
    assert "DLL load failed: a dependency is missing" in proc.stdout
