import subprocess
import sys

import divsel


def test_import():
    assert divsel.__version__ == "0.1.0"


def test_missing_extension_points_at_maturin_develop():
    """The ``except ImportError`` arm of ``python/divsel/__init__.py``.

    It is the first thing a source-checkout user hits, and nothing exercised it:
    the extension is always present in a venv where the suite can run. A meta
    path finder that refuses ``divsel._divsel`` reproduces the state without
    touching the installation, in a child process so this one keeps its import.
    """
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'divsel._divsel':\n"
        "            raise ImportError('no compiled extension in this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "try:\n"
        "    import divsel\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('divsel imported without its extension module')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "maturin develop" in proc.stdout
    assert "_divsel" in proc.stdout
