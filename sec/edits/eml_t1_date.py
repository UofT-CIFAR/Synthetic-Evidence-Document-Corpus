"""X-T1-DATE-EML: change visible Date header on a rendered email (spec §6).

Mirrors receipt Tier-1 flow: render native pixels from the source message, then
one full-frame vision edit so variants A–D still exercise the image stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

from email.message import EmailMessage
from email.utils import format_datetime, parsedate_to_datetime
from PIL import Image

from ..adapters.base import AdapterCapabilityError, AdapterCredentialError, VariantAdapter
from ..email_render import baseline_email_rgb, render_email_png
from ..ocr.image_date_bbox import primary_date_on_image
from ..sources.mail_base import EmailItem, load_email_bytes, parse_email_message
from .common import apply_full_image_inpaint


# Uniform offsets per spec §6 X-T1-DATE-EML (±1h, ±6h, ±1d, ±7d, ±30d).
_OFFSETS: tuple[timedelta, ...] = (
    timedelta(hours=-1),
    timedelta(hours=-6),
    timedelta(days=-1),
    timedelta(days=-7),
    timedelta(days=-30),
    timedelta(hours=1),
    timedelta(hours=6),
    timedelta(days=1),
    timedelta(days=7),
    timedelta(days=30),
)


def _offset_for(item_index: int) -> timedelta:
    return _OFFSETS[item_index % len(_OFFSETS)]


def _spec_audit_block(
    *,
    bbox: tuple[int, int, int, int],
    old_date: str,
    new_date: str,
    coherent_received: bool,
    prompts_dir: Path,
) -> str:
    path = prompts_dir / "T1-EML-DATE.md"
    if path.exists():
        return (
            path.read_text(encoding="utf-8")
            .replace("{bbox}", str(bbox))
            .replace("{old_date}", old_date)
            .replace("{new_date}", new_date)
            .replace("{coherent_received}", "true" if coherent_received else "false")
        )
    return (
        "X-T1-DATE-EML — Change a date on an email.\n"
        f"(Pre-located bbox={bbox}, old_date={old_date!r}, new_date={new_date!r}, "
        f"coherent_received={coherent_received})\n"
    )


def _inpaint_prompt(
    *,
    old_date: str,
    new_date: str,
    coherent_received: bool,
) -> str:
    recv_clause = (
        "Also adjust any visible **Received:** hop lines so their embedded times stay "
        "plausible relative to the new Date (consistent threading)."
        if coherent_received
        else (
            "Change **only** the **Date:** line. Leave every **Received:** line "
            "exactly as in the input (amateur inconsistency)."
        )
    )
    return (
        "Recreate this email screenshot so it matches the input pixel-for-pixel (layout, "
        "fonts, margins, paper tone) except update the mail headers block: "
        f'replace the Date value "{old_date}" with "{new_date}". '
        f"{recv_clause} "
        "Do not alter the message body wording. No borders or watermarks. Output one RGB "
        "image with exactly the same width and height as the input."
    )


@dataclass
class EmlDateResult:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    old_date: str
    new_date: str
    coherent_received: bool
    prompt: str
    notes: str = ""


def _normalize_msg_date(msg: EmailMessage) -> tuple[str, object]:
    raw = msg.get("Date")
    if not raw:
        raise ValueError("missing Date header")
    raw = raw.strip()
    dt = parsedate_to_datetime(raw)
    if dt is None:
        raise ValueError("unparseable Date header")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    new_hdr = format_datetime(dt)
    return raw, dt


def apply(
    item: EmailItem,
    *,
    adapter: VariantAdapter,
    item_index: int,
    seed: int,
    prompts_dir: Path,
) -> EmlDateResult | None:
    delta = _offset_for(item_index)
    coherent_received = item_index % 2 == 0

    if getattr(item, "modality", "rfc822") == "rvl_email_page":
        base_img = baseline_email_rgb(item)
        hit = primary_date_on_image(base_img)
        if hit is None:
            return None
        date_text, base_dt, fmt, bbox = hit
        new_dt = base_dt + delta
        new_date = new_dt.strftime(fmt)
        old_date = date_text
    else:
        data = load_email_bytes(item.path)
        msg = parse_email_message(data)
        old_date, base_dt = _normalize_msg_date(msg)
        new_dt = base_dt + delta
        new_date = format_datetime(new_dt)
        msg_date_hdr = msg.get("Date", "").strip()
        date_text = msg_date_hdr or old_date

        include_rx = 2 if msg.get_all("Received") else 0
        base_img, meta = render_email_png(msg, include_received=include_rx)
        bbox = meta.get("date_bbox")
        if meta.get("date_text"):
            date_text = meta["date_text"]
        if not bbox:
            w, h = base_img.size
            bbox = (int(w * 0.04), int(h * 0.04), int(w * 0.72), 26)

    api_prompt = _inpaint_prompt(
        old_date=date_text,
        new_date=new_date,
        coherent_received=coherent_received,
    )
    audit = _spec_audit_block(
        bbox=bbox,
        old_date=date_text,
        new_date=new_date,
        coherent_received=coherent_received,
        prompts_dir=prompts_dir,
    )
    prompt = (
        "### API inpaint — full frame (email screenshot; Date / Received policy)\n"
        f"{api_prompt}\n\n"
        "### Spec procedure (audit / manifest)\n"
        f"{audit}"
    )
    try:
        out = apply_full_image_inpaint(base_img.convert("RGB"), adapter=adapter, prompt=api_prompt, seed=seed)
    except AdapterCredentialError:
        raise
    except AdapterCapabilityError:
        return None

    return EmlDateResult(
        image=out.convert("RGB"),
        bbox=bbox,
        old_date=date_text,
        new_date=new_date,
        coherent_received=coherent_received,
        prompt=prompt,
        notes=(
            f"coherent_received={coherent_received}; delta={delta}; edit_path=api_full_image; "
            f"modality={getattr(item, 'modality', 'rfc822')}"
        ),
    )
