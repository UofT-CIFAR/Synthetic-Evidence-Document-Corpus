# ComfyUI (variant D) setup for the Synthetic Evidence Corpus

The pipeline calls a **running** ComfyUI server over HTTP (`COMFYUI_URL`). Workflows are **not** stored inside ComfyUI’s UI; they live here as JSON and are submitted via `/prompt`.

## Quick checklist

1. **Install ComfyUI** from [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) (Python venv + `pip install -r requirements.txt`).
2. **Log in to Hugging Face** and accept the licenses for:
   - [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) (gated)
   - [stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) (gated)
3. **Download checkpoints** into `ComfyUI/models/checkpoints/` with the **exact filenames** expected by the JSON (or edit `ckpt_name` in the JSON):

   | File | Role |
   |------|------|
   | `flux1-fill-dev.safetensors` | Flux.1 Fill — inpainting (T1–T3, variant D) |
   | `sd3_medium_incl_clips_t5xxlfp8.safetensors` | SD3 Medium — T2I (T4, variant D) |

   Helper (requires `huggingface-cli login`):

   ```bash
   export COMFYUI_ROOT=~/ComfyUI   # your install path
   ./scripts/download_comfyui_checkpoints.sh
   ```

4. **Start ComfyUI** (default port 8188):

   ```bash
   cd /path/to/ComfyUI && python main.py
   ```

5. **Point the SEC project at the server**:

   ```bash
   export COMFYUI_URL=http://127.0.0.1:8188
   ```

6. **Verify**:

   ```bash
   python -m scripts.verify_comfyui --comfy-root "$COMFYUI_ROOT"
   ```

7. **Optional one-liner** (prints reminders + ensures `requests`):

   ```bash
   ./scripts/setup_comfyui_for_sec.sh
   ```

## Workflow files

| File | Adapter key | Used for |
|------|-------------|----------|
| `flux_fill_inpaint.json` | `inpaint` | Masked edits (T1–T3) |
| `sd3_medium_generate.json` | `generate` | Full image gen (T4) |

Placeholders `<<INPUT_IMAGE>>`, `<<INPUT_MASK>>`, `<<PROMPT>>`, `<<SEED>>`, `<<WIDTH>>`, `<<HEIGHT>>` are replaced at run time; do not remove them.

## If nodes fail in your ComfyUI version

Graphs use common node types (`LoadImage`, `KSampler`, etc.). If your build uses different **class_type** names or Flux is loaded from `models/diffusion_models/`, export a **working** graph from the ComfyUI UI into these files and **keep the same `<<...>>` placeholders** (or copy the server’s API JSON from “Save (API format)”).

## 403 / `GatedRepoError` on download

Both official checkpoints are **gated** on Hugging Face. A **403** means the CLI is not allowed to read the file yet.

1. In the **browser**, while logged into the **same** Hugging Face account you will use in the terminal:
   - [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) — click **Agree** / **Request access** (wait if access is manual).
   - [SD3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) — same.
2. In the terminal: `hf auth login` (or `huggingface-cli login`) and paste a token with **Read** from [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens).

### Fine-grained token: “public gated repositories”

If the error says **“Please enable access to public gated repositories in your fine-grained token settings”**, your token is **fine-grained** and blocked from gated model repos. Fix it in one of these ways:

- **Edit the token** on the tokens page, find **Repository access** / **Gated** / **Public gated repositories**, and **enable access to public gated repositories** (wording can vary by HF UI version), *or*
- Create a new **Classic** token with **Read** (classic tokens can read gated repos you’ve accepted in the browser—no extra “gated” toggle).

Then `hf auth login` again.

3. Run `./scripts/download_comfyui_checkpoints.sh` again.

## Text for T3 / T4

`tools.yaml` sets `text_fallback: A` for variant D: **Tier 3 / Tier 4 language** uses the **OpenAI** adapter. Set `OPENAI_API_KEY` even when using ComfyUI for images.
