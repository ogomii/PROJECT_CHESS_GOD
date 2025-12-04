#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory (works when invoked from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow overriding Python executable via environment variable
PYTHON=${PYTHON:-python3}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: Python not found: $PYTHON" >&2
  exit 1
fi

# Ensure project root is on PYTHONPATH so sibling packages (like `engines`) are importable
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Execute the project's main script
exec "$PYTHON" "$SCRIPT_DIR/src/main.py" "$@"
