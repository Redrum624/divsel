#!/usr/bin/env bash
# One installability cell: a fresh venv, `pip install <spec>`, `python -c "import <module>"`,
# and a JSON record with ok/fail plus the tail of pip's output.
#
#   install_cell.sh <library> <pip spec> <import module> <out dir>
#
# Never exits non-zero for an install failure -- the failure IS the measurement. The calling
# workflow step still carries `continue-on-error: true` so an unexpected script error cannot
# stop the other cells either.
set -u
LIB="$1"; SPEC="$2"; MODULE="$3"; OUT="$4"
mkdir -p "$OUT"
OS_NAME="${RUNNER_OS:-$(uname -s)}"
PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_FULL="$(python -c 'import sys, platform; print(sys.version.split()[0], platform.machine())')"
VENV="venv-$LIB"
rm -rf "$VENV"
python -m venv "$VENV"
# The venv is scratch, and it is created in the checkout root -- where the divsel
# cell then runs `pip install .`. Remove it on the way out however this exits, so
# a local run does not leave four of them (hundreds of MB) behind, and so no cell
# builds a source tree with another cell's venv sitting in it.
trap 'rm -rf "$VENV"' EXIT
if [ -x "$VENV/Scripts/python.exe" ]; then VPY="$VENV/Scripts/python.exe"; else VPY="$VENV/bin/python"; fi
"$VPY" -m pip install --upgrade pip > /dev/null 2>&1 || true

LOG="$OUT/$LIB.log"
echo "\$ pip install $SPEC" > "$LOG"
# shellcheck disable=SC2086
"$VPY" -m pip install $SPEC >> "$LOG" 2>&1
INSTALL_RC=$?
IMPORT_RC=1
IMPORT_MSG=""
if [ "$INSTALL_RC" -eq 0 ]; then
  IMPORT_MSG="$("$VPY" -c "import $MODULE, importlib.metadata as m; print('import ok;', '$LIB', m.version('$LIB'))" 2>&1)"
  IMPORT_RC=$?
  echo "\$ python -c \"import $MODULE\"" >> "$LOG"
  echo "$IMPORT_MSG" >> "$LOG"
fi
if [ "$INSTALL_RC" -eq 0 ] && [ "$IMPORT_RC" -eq 0 ]; then STATUS=ok; else STATUS=fail; fi

python - "$OUT" "$LIB" "$SPEC" "$MODULE" "$OS_NAME" "$PY_VER" "$PY_FULL" "$STATUS" "$INSTALL_RC" "$IMPORT_RC" "$LOG" <<'EOF'
import json, sys
out, lib, spec, module, os_name, py, py_full, status, install_rc, import_rc, log = sys.argv[1:]
lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
rec = {
    "library": lib, "spec": spec, "module": module, "os": os_name, "python": py, "python_full": py_full,
    "status": status, "install_rc": int(install_rc), "import_rc": int(import_rc),
    "first_line": next((l for l in lines[1:] if l.strip()), ""),
    "tail": lines[-15:],
}
name = f"{os_name}-py{py}-{lib}.json".replace(" ", "_")
json.dump(rec, open(f"{out}/{name}", "w", encoding="utf-8"), indent=1)
print(f"{lib}: {status} (pip rc={install_rc}, import rc={import_rc})")
print("\n".join("    " + l for l in lines[-6:]))
EOF
exit 0
