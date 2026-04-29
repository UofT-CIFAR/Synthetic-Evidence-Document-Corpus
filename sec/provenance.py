"""Provenance marking (spec §2 rule 3).

Every file written to the corpus must carry at least one provenance marker:
1. EXIF UserComment tag (JPEG) or PNG text chunk equivalent.
2. XMP sidecar metadata with a `sec:synthetic` property.
3. A 64-bit LSB steganographic magic number in a fixed pixel region, which
   survives at least one round-trip through a consumer image editor.

We also record the SHA-256 of the file both before and after marker injection
in the manifest. The acceptance check in `verify_marker` considers an artifact
provenance-valid if ANY of the three markers is present.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import piexif
from PIL import Image, PngImagePlugin

MAGIC_HEX_DEFAULT = "5345432d53594e5448"  # "SEC-SYNTH"
EXIF_COMMENT_DEFAULT = "synthetic-evidence-corpus:true"
XMP_NS_DEFAULT = "https://synthetic-evidence-corpus.example/ns/1.0"


@dataclass(frozen=True)
class ProvenanceConfig:
    exif_comment: str = EXIF_COMMENT_DEFAULT
    xmp_namespace: str = XMP_NS_DEFAULT
    xmp_key: str = "synthetic"
    magic_hex: str = MAGIC_HEX_DEFAULT
    stego_x0: int = 0
    stego_y0: int = 0


def sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex_to_bits(hex_str: str) -> list[int]:
    raw = bytes.fromhex(hex_str)
    bits: list[int] = []
    for byte in raw:
        for bit_position in range(7, -1, -1):
            bits.append((byte >> bit_position) & 1)
    return bits


def _bits_to_hex(bits: list[int]) -> str:
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | (b & 1)
        out.append(byte)
    return out.hex()


def _embed_stego(image: Image.Image, cfg: ProvenanceConfig) -> Image.Image:
    """Embed the magic number as LSBs on the first channel of a horizontal run."""

    img = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image.copy()
    bits = _hex_to_bits(cfg.magic_hex)
    width, _ = img.size
    if cfg.stego_x0 + len(bits) > width:
        # Fall back to wrapping across rows if image is narrower than 72 px.
        wrap = True
    else:
        wrap = False
    pixels = img.load()
    assert pixels is not None
    x, y = cfg.stego_x0, cfg.stego_y0
    for bit in bits:
        px = pixels[x, y]
        first = (px[0] & ~1) | bit
        pixels[x, y] = (first,) + tuple(px[1:])
        x += 1
        if wrap and x >= width:
            x = 0
            y += 1
    return img


def _extract_stego(image: Image.Image, cfg: ProvenanceConfig) -> str:
    bits_needed = len(cfg.magic_hex) * 4
    img = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image
    pixels = img.load()
    assert pixels is not None
    width, _ = img.size
    x, y = cfg.stego_x0, cfg.stego_y0
    bits: list[int] = []
    for _ in range(bits_needed):
        px = pixels[x, y]
        bits.append(px[0] & 1)
        x += 1
        if x >= width:
            x = 0
            y += 1
    return _bits_to_hex(bits)


def _xmp_packet(cfg: ProvenanceConfig) -> str:
    return (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        f'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:sec="{cfg.xmp_namespace}">'
        f'<rdf:Description rdf:about=""><sec:{cfg.xmp_key}>true</sec:{cfg.xmp_key}></rdf:Description>'
        "</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    )


def write_image_with_provenance(
    image: Image.Image,
    out_path: Path,
    cfg: ProvenanceConfig | None = None,
) -> dict:
    """Write the image with stego + format-native metadata markers.

    Returns a manifest-ready dict with `sha256_pre_marker`, `sha256`,
    `provenance_marker` (compact JSON description).
    """

    cfg = cfg or ProvenanceConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Hash of the un-marked image (as PNG bytes, format-independent).
    pre_bytes_io = _to_png_bytes(image)
    sha_pre = sha256_of_bytes(pre_bytes_io)

    stamped = _embed_stego(image, cfg)

    suffix = out_path.suffix.lower()
    exif_status = "skipped"
    xmp_status = "skipped"

    if suffix in (".jpg", ".jpeg"):
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
            b"ASCII\0\0\0" + cfg.exif_comment.encode("ascii", errors="ignore")
        )
        exif_bytes = piexif.dump(exif_dict)
        stamped.save(out_path, format="JPEG", quality=95, exif=exif_bytes)
        exif_status = "written"
    elif suffix == ".png":
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Comment", cfg.exif_comment)
        pnginfo.add_text("XMP", _xmp_packet(cfg))
        stamped.save(out_path, format="PNG", pnginfo=pnginfo)
        xmp_status = "written"
        exif_status = "written"  # stored as tEXt chunk
    else:
        stamped.save(out_path)

    sha_post = sha256_of_file(out_path)
    marker = {
        "stego_magic": cfg.magic_hex,
        "exif": exif_status,
        "xmp": xmp_status,
        "sha256_pre_marker": sha_pre,
    }
    return {
        "provenance_marker": json.dumps(marker, separators=(",", ":")),
        "sha256": sha_post,
        "sha256_pre_marker": sha_pre,
    }


def _to_png_bytes(image: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def verify_marker(path: Path, cfg: ProvenanceConfig | None = None) -> bool:
    """Return True if any of the markers is detected in the file."""

    cfg = cfg or ProvenanceConfig()
    path = Path(path)
    if not path.exists():
        return False
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return False

    if _extract_stego(img, cfg) == cfg.magic_hex:
        return True

    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        try:
            exif = piexif.load(str(path))
            uc = exif.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
            if cfg.exif_comment.encode("ascii", errors="ignore") in uc:
                return True
        except Exception:
            pass
    elif suffix == ".png":
        info = img.info
        if cfg.exif_comment in (info.get("Comment") or ""):
            return True
        if cfg.xmp_namespace in (info.get("XMP") or ""):
            return True
    return False
