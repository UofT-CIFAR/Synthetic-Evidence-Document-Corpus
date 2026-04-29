#!/usr/bin/env bash
# Download variant-D checkpoints into ComfyUI models/checkpoints (optional).
#
# Prerequisites:
#   pip install -U huggingface_hub
#   hf auth login
#     (or: huggingface-cli login)
#   Use a token with *Read* access: https://huggingface.co/settings/tokens
#
# GATED REPOS (403 = accept license in browser + token can read gated repos):
#   1) Open https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
#      — Agree / Request access; wait for approval if required
#   2) Open https://huggingface.co/stabilityai/stable-diffusion-3-medium — same
#   3) Fine-grained tokens: enable "access to public gated repositories" on the token,
#      OR use a Classic token with Read — see configs/comfyui/README.md
#   The CLI must use the *same* HF account you used in the browser.
#
# Usage:
#   export COMFYUI_ROOT=~/ComfyUI
#   ./scripts/download_comfyui_checkpoints.sh
#
# Models (large; ~24GB + ~5GB typical):
#   - black-forest-labs/FLUX.1-Fill-dev::flux1-fill-dev.safetensors
#   - stabilityai/stable-diffusion-3-medium::sd3_medium_incl_clips_t5xxlfp8.safetensors
#
set -euo pipefail

COMFYUI_ROOT="${COMFYUI_ROOT:-$HOME/ComfyUI}"
DEST="${COMFYUI_ROOT}/models/checkpoints"
mkdir -p "$DEST"

HF_DL=()
if command -v hf &>/dev/null; then
  HF_DL=(hf download)
elif command -v huggingface-cli &>/dev/null; then
  HF_DL=(huggingface-cli download)
else
  echo "Install: pip install -U huggingface_hub" >&2
  echo "Then: hf auth login" >&2
  exit 1
fi

echo "Downloading into: $DEST"
echo ""
echo "If you get 403 GatedRepoError: open each model page in a browser, accept the license"
echo "for the same account you use with 'hf auth login', then run this script again."
echo ""

# Prefer modern CLI (no deprecated warning)
if [[ "${HF_DL[0]}" == "hf" ]]; then
  "${HF_DL[@]}" black-forest-labs/FLUX.1-Fill-dev flux1-fill-dev.safetensors --local-dir "$DEST"
  "${HF_DL[@]}" stabilityai/stable-diffusion-3-medium sd3_medium_incl_clips_t5xxlfp8.safetensors --local-dir "$DEST"
else
  huggingface-cli download black-forest-labs/FLUX.1-Fill-dev \
    flux1-fill-dev.safetensors \
    --local-dir "$DEST" --local-dir-use-symlinks False
  huggingface-cli download stabilityai/stable-diffusion-3-medium \
    sd3_medium_incl_clips_t5xxlfp8.safetensors \
    --local-dir "$DEST" --local-dir-use-symlinks False
fi

echo "Done. Filenames must match configs/comfyui/*.json (ckpt_name)."
