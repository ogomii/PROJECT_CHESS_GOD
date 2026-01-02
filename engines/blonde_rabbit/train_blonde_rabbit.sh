#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory (works when invoked from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Root directory is two levels up from blonde_rabbit
ROOT_DIR="$(cd "$SCRIPT_DIR/../../" && pwd)"

# Allow overriding Python executable via environment variable
PYTHON=${PYTHON:-python3}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: Python not found: $PYTHON" >&2
  exit 1
fi

# Ensure project root is on PYTHONPATH so packages like 'tools' are importable
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Execute train.py from the root directory
exec "$PYTHON" "$ROOT_DIR/engines/blonde_rabbit/src/train.py" "$@"