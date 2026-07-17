#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-"$ROOT/docs/case-studies/module3-dit-math-derivation.md"}"
OUTPUT="${2:-"$ROOT/output/pdf/module3-dit-math-derivation.pdf"}"

mkdir -p "$(dirname "$OUTPUT")"

if [ ! -d "$ROOT/scripts/node_modules" ]; then
  echo "[setup] Installing renderer dependencies in scripts/..."
  npm --prefix "$ROOT/scripts" install
fi

node "$ROOT/scripts/render_pdf.js" "$INPUT" "$OUTPUT"

