#!/usr/bin/env bash
# One-time / repeat setup to run ComfyUI with the Synthetic Evidence Corpus
# variant D (Flux.1-Fill + SD3 Medium) workflows.
#
# You still need to:
#   1) Clone/install ComfyUI from https://github.com/comfyanonymous/ComfyUI
#   2) Create a venv and: pip install -r ComfyUI/requirements.txt
#   3) Log in to Hugging Face and accept licenses for gated models
#   4) Run scripts/download_comfyui_checkpoints.sh (or place .safetensors manually)
#   5) Start ComfyUI:  cd ComfyUI && python main.py
#   6) export COMFYUI_URL=http://127.0.0.1:8188
#   7) python -m scripts.verify_comfyui --comfy-root "$COMFYUI_ROOT"
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> SEC project: $ROOT"
echo "==> Installing Python deps for ComfyUI adapter (requests) if needed"
python3 -m pip install -q 'requests>=2.31' || true

echo "==> Sample environment (add to ~/.bashrc or run before batches):"
echo "    export COMFYUI_URL=http://127.0.0.1:8188"
echo ""
echo "==> Verify API (start ComfyUI first):"
echo "    export COMFYUI_URL=... && python3 -m scripts.verify_comfyui --comfy-root \"\${COMFYUI_ROOT:-\$HOME/ComfyUI}\""
echo ""
echo "==> Workflows used by the code (do not move):"
echo "    $ROOT/configs/comfyui/flux_fill_inpaint.json"
echo "    $ROOT/configs/comfyui/sd3_medium_generate.json"
echo ""
echo "If node types do not match your ComfyUI version, export a working graph from the"
echo "ComfyUI UI to these files, keeping placeholder strings <<INPUT_IMAGE>>, <<INPUT_MASK>>,"
echo "<<PROMPT>>, <<SEED>>, <<WIDTH>>, <<HEIGHT>> in the JSON."
echo ""
echo "Done."
