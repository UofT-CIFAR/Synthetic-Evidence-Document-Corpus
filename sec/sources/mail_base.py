"""Shared types for email-source loaders (EML family)."""

from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path


@dataclass(frozen=True)
class EmailItem:
    """RFC822 ``.eml`` file or cached **RVL-CDIP email** page raster."""

    doc_id: str
    path: Path
    pool_hint: str = "train"
    modality: str = "rfc822"  # 'rfc822' | 'rvl_email_page'


def load_email_bytes(path: Path) -> bytes:
    return path.read_bytes()


def parse_email_message(data: bytes) -> EmailMessage:
    parser = BytesParser(policy=policy.default)
    msg = parser.parsebytes(data)
    if not isinstance(msg, EmailMessage):
        raise ValueError("Parsed message is not EmailMessage")
    return msg


def decode_mail_payload(data: bytes, charset: str | None) -> str:
    """Decode message bytes using MIME charset; fall back when codec name is unknown.

    Some MUAs emit ``windows-874`` etc.; Python expects ``cp874``.
    """

    raw = (charset or "").strip() or "utf-8"
    low = raw.lower().replace("_", "-")

    if low in ("unknown-8bit", "binary", "x-unknown"):
        return data.decode("utf-8", errors="replace")

    aliases: dict[str, str] = {
        "windows-874": "cp874",
        "windows-1250": "cp1250",
        "windows-1251": "cp1251",
        "windows-1252": "cp1252",
        "windows-1253": "cp1253",
        "windows-1254": "cp1254",
        "windows-1255": "cp1255",
        "windows-1256": "cp1256",
        "windows-1257": "cp1257",
        "windows-1258": "cp1258",
    }
    ordered = []
    if low in aliases:
        ordered.append(aliases[low])
    ordered.append(raw)
    # Duplicate strip like windows-* → cp* generic
    if low.startswith("windows-") and low[8:].isdigit():
        cp = f"cp{low[8:]}"
        if cp not in ordered:
            ordered.insert(0, cp)

    for enc in ordered:
        try:
            return data.decode(enc, errors="replace")
        except LookupError:
            continue
    return data.decode("utf-8", errors="replace")


def plain_body(msg: EmailMessage) -> str:
    """Best-effort plain text body for prompting / eligibility."""

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    return decode_mail_payload(payload, part.get_content_charset())
                return str(payload or "")
        return ""
    if msg.get_content_type() == "text/plain":
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            return decode_mail_payload(payload, msg.get_content_charset())
        return str(payload or "")
    return ""


def plain_body_from_item(item: EmailItem) -> str:
    """Plain-ish text for prompts / eligibility (RFC822 body or OCR on page images)."""

    if getattr(item, "modality", "rfc822") != "rvl_email_page":
        try:
            return plain_body(parse_email_message(load_email_bytes(item.path)))
        except Exception:
            return ""
    try:
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(item.path).convert("RGB"))
    except Exception:
        return ""
