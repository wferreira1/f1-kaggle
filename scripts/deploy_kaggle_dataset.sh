#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
DATASET_ID="${2:-}"
TITLE="${3:-F1 Kaggle Project Code}"
VERSION_MESSAGE="${4:-Update project code}"

if [[ "$MODE" != "create" && "$MODE" != "version" ]]; then
  echo "Uso: $0 <create|version> <username/dataset-slug> [title] [version-message]"
  exit 1
fi

if [[ -z "$DATASET_ID" ]]; then
  echo "Informe o dataset id no formato username/dataset-slug."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/.kaggle-build/dataset"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp "$ROOT_DIR/README.md" "$BUILD_DIR/"
cp "$ROOT_DIR/pyproject.toml" "$BUILD_DIR/"
cp "$ROOT_DIR/uv.lock" "$BUILD_DIR/"
find "$ROOT_DIR/src" -maxdepth 1 -type f \( -name "*.py" -o -name "*.sh" \) -exec cp {} "$BUILD_DIR/" \;

cat > "$BUILD_DIR/dataset-metadata.json" <<EOF
{
  "title": "$TITLE",
  "id": "$DATASET_ID",
  "licenses": [
    {
      "name": "other"
    }
  ]
}
EOF

if [[ "$MODE" == "create" ]]; then
  kaggle datasets create -p "$BUILD_DIR"
else
  kaggle datasets version -p "$BUILD_DIR" -m "$VERSION_MESSAGE"
fi
