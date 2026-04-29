"""Variant D: local ComfyUI with Flux.1-Fill (T1/T2/T3 inpaint) and SD3 Medium (T4).

Calls ComfyUI's HTTP API. Workflow JSONs live under ``configs/comfyui/``.
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from .base import AdapterCapabilityError, AdapterCredentialError, AdapterInfo


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "comfyui"


class ComfyUIAdapter:
    def __init__(self, variant_cfg: dict[str, Any], tools_cfg: dict[str, Any] | None = None) -> None:
        url_env = variant_cfg.get("env", {}).get("url", "COMFYUI_URL")
        self._base_url = os.environ.get(url_env) or variant_cfg.get("endpoint")
        self._workflows = variant_cfg.get("workflows", {})
        self._tools_cfg = tools_cfg or {}
        self._text_fallback = variant_cfg.get("text_fallback", "A")
        self.info = AdapterInfo(
            variant="D",
            tool_family=variant_cfg.get("tool_family", "open_llm"),
            tool_specific=variant_cfg.get("tool_specific", "comfyui:flux+sd3"),
        )

    def _require_url(self) -> str:
        if not self._base_url:
            raise AdapterCredentialError(
                "Variant D requires COMFYUI_URL in the environment (e.g. http://localhost:8188)."
            )
        return self._base_url.rstrip("/")

    # --- helpers -----------------------------------------------------------

    def _load_workflow(self, key: str) -> dict:
        name = self._workflows.get(key)
        if not name:
            raise AdapterCapabilityError(f"ComfyUI workflow {key!r} not configured")
        path = CONFIG_DIR / name
        if not path.exists():
            raise AdapterCapabilityError(f"ComfyUI workflow file missing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            w = json.load(f)
        w.pop("_comment", None)
        return w

    def _upload_image(self, image: Image.Image, filename: str) -> str:
        import requests

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        resp = requests.post(
            f"{self._require_url()}/upload/image",
            files={"image": (filename, buf, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["name"]

    def _queue_prompt(self, workflow: dict) -> str:
        import requests

        client_id = uuid.uuid4().hex
        payload = {"prompt": workflow, "client_id": client_id}
        resp = requests.post(
            f"{self._require_url()}/prompt",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def _await_result(self, prompt_id: str, timeout: float = 600.0) -> dict:
        import requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(f"{self._require_url()}/history/{prompt_id}", timeout=30)
            if resp.ok:
                payload = resp.json()
                if prompt_id in payload and payload[prompt_id].get("outputs"):
                    return payload[prompt_id]
            time.sleep(1.5)
        raise RuntimeError(f"ComfyUI prompt {prompt_id} did not finish within {timeout}s")

    def _fetch_output_image(self, history: dict) -> Image.Image:
        import requests

        for node_id, node_out in history.get("outputs", {}).items():
            images = node_out.get("images") or []
            for img_meta in images:
                params = {
                    "filename": img_meta["filename"],
                    "subfolder": img_meta.get("subfolder", ""),
                    "type": img_meta.get("type", "output"),
                }
                resp = requests.get(
                    f"{self._require_url()}/view", params=params, timeout=60
                )
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        raise RuntimeError("ComfyUI history contained no output images")

    @staticmethod
    def _apply_substitutions(workflow: dict, substitutions: dict[str, Any]) -> dict:
        """Replace placeholder strings ``<<KEY>>`` in the workflow with real values.

        Templates store placeholders as JSON string values, e.g. ``"image": \"<<INPUT_IMAGE>>\"``
        (loaded as the Python str ``\"<<INPUT_IMAGE>>\"``) or similar; this walks the
        tree and replaces any string exactly equal to ``<<KEY>>`` with ``substitutions[KEY]``,
        preserving correct types (int for seed, str for image names, etc.).
        """

        def _sub(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sub(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sub(x) for x in obj]
            if isinstance(obj, str) and len(obj) >= 4 and obj.startswith("<<") and obj.endswith(">>"):
                k = obj[2:-2]
                if k in substitutions:
                    return substitutions[k]
            return obj

        return _sub(workflow)

    # --- capabilities ------------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        workflow = self._load_workflow("inpaint")
        img_name = self._upload_image(image, f"sec_src_{seed}.png")
        mask_name = self._upload_image(mask.convert("L"), f"sec_mask_{seed}.png")
        workflow = self._apply_substitutions(
            workflow,
            {
                "INPUT_IMAGE": img_name,
                "INPUT_MASK": mask_name,
                "PROMPT": prompt,
                "SEED": int(seed) & 0xFFFFFFFF,
            },
        )
        prompt_id = self._queue_prompt(workflow)
        history = self._await_result(prompt_id)
        return self._fetch_output_image(history)

    def few_shot_image(
        self,
        refs: list[Image.Image],
        prompt: str,
        seed: int,
        size: tuple[int, int] = (512, 512),
    ) -> Image.Image:
        workflow = self._load_workflow("generate")
        # SD3 T2I template in ``sd3_medium_generate.json`` is text+latent; optional
        # ``<<REF_0>>``… nodes can be added for IP-Adapter–style use. We still upload
        # refs for forward compatibility.
        for i, ref in enumerate(refs[:4]):
            self._upload_image(ref, f"sec_ref_{seed}_{i}.png")
        workflow = self._apply_substitutions(
            workflow,
            {
                "PROMPT": prompt,
                "SEED": int(seed) & 0xFFFFFFFF,
                "WIDTH": int(size[0]),
                "HEIGHT": int(size[1]),
            },
        )
        prompt_id = self._queue_prompt(workflow)
        history = self._await_result(prompt_id)
        return self._fetch_output_image(history)

    def text_complete(self, prompt: str, seed: int, max_tokens: int = 400) -> str:
        # ComfyUI is image-only in this pipeline; delegate to the configured
        # text fallback.
        from .base import load_adapter

        fallback = load_adapter(self._text_fallback, self._tools_cfg)
        return fallback.text_complete(prompt, seed, max_tokens=max_tokens)
