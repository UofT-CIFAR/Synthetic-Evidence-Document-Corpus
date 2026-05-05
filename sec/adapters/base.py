"""Common adapter contract for all four variants (spec §5).

Every variant exposes three operations so tier code is variant-agnostic:

- ``inpaint``: edit a region of an existing image given a binary mask.
- ``few_shot_image``: produce a new image conditioned on reference images and
  a prompt (used for Tier-2 signatures and Tier-4 fabrication).
- ``text_complete``: return a string (used for Tier-3 insertion drafting).

Adapters that cannot fulfil a particular method raise
``AdapterCapabilityError`` or ``AdapterCredentialError``.

RCT tier pipelines (batch runner) do **not** synthesize corpus pixels locally
when vision or text calls fail, except Tier-1 date with explicit
``tier1_date.use_local_burn_only: true``. Some adapters may delegate
``text_complete`` to another configured variant (e.g. Ideogram → OpenAI); that
is still an API call, not a local corpus fallback.

Style-pool bootstrap (`scripts.phase0_setup`) uses deterministic PIL strokes
only when **no** image ``adapter`` is passed to ``populate_pools``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from PIL import Image


class AdapterCapabilityError(RuntimeError):
    """Raised when an adapter cannot satisfy a requested capability."""


class AdapterCredentialError(RuntimeError):
    """Raised when required credentials (API key, URL) are absent."""


@dataclass
class AdapterInfo:
    variant: str
    tool_family: str
    tool_specific: str


@runtime_checkable
class VariantAdapter(Protocol):
    info: AdapterInfo

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image: ...

    def few_shot_image(
        self,
        refs: list[Image.Image],
        prompt: str,
        seed: int,
        size: tuple[int, int] = (512, 512),
    ) -> Image.Image: ...

    def text_complete(self, prompt: str, seed: int, max_tokens: int = 400) -> str: ...


def load_adapter(variant: str, tools_cfg: dict[str, Any]) -> VariantAdapter:
    """Instantiate the adapter named by ``tools.yaml`` for the given variant."""

    variant = variant.upper()
    variants_cfg = tools_cfg.get("variants", {})
    if variant not in variants_cfg:
        raise KeyError(f"Variant {variant!r} not configured in tools.yaml")
    vcfg = variants_cfg[variant]
    module_path, _, class_name = vcfg["adapter"].partition(":")
    import importlib

    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    adapter = adapter_cls(vcfg, tools_cfg=tools_cfg)
    if not isinstance(adapter, VariantAdapter):  # runtime protocol check
        raise TypeError(f"Adapter {class_name} does not implement VariantAdapter")
    return adapter
