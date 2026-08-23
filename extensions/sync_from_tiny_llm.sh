#!/bin/sh
# Re-sync this mirror from the live course checkout after kernel edits.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-$DIR/../tiny-llm/src/extensions}"
for item in CMakeLists.txt bindings.cpp build.py test.py src tiny_llm_ext; do
    rm -rf "$DIR/$item"
    cp -r "$SRC/$item" "$DIR/$item"
done
find "$DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "mirror updated from $SRC"
