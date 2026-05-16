#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="$ROOT_DIR/kaggle/kernel"

if [[ ! -f "$KERNEL_DIR/kernel-metadata.json" ]]; then
  echo "Arquivo ausente: $KERNEL_DIR/kernel-metadata.json"
  echo "Copie kernel-metadata.json.example para kernel-metadata.json e preencha seus ids do Kaggle."
  exit 1
fi

kaggle kernels push -p "$KERNEL_DIR"
