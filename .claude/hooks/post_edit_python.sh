#!/usr/bin/env bash
# PostToolUse hook: ruff check + pytest for edited Python files

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only act on Python files
[[ "$FILE_PATH" == *.py ]] || exit 0

echo "--- ruff check ---"
ruff check "$FILE_PATH" || true

# Determine which test file to run
TEST_FILE=""

if [[ "$(basename "$FILE_PATH")" == test_* ]]; then
  # The edited file is already a test file
  TEST_FILE="$FILE_PATH"
else
  # Look for a matching test file: src/.../foo.py → tests/**/test_foo.py
  MODULE_NAME="$(basename "$FILE_PATH" .py)"
  PROJECT_ROOT="$(git -C "$(dirname "$FILE_PATH")" rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$FILE_PATH")")"
  TEST_FILE=$(find "$PROJECT_ROOT/tests" -name "test_${MODULE_NAME}.py" 2>/dev/null | head -1 || true)
fi

if [[ -n "$TEST_FILE" && -f "$TEST_FILE" ]]; then
  echo "--- pytest $TEST_FILE ---"
  pytest "$TEST_FILE" -q || true
fi

exit 0
