"""Rasterize PDF pages to PNG for DOC loaders (DUDE, UCSF)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def render_pdf_first_page_to_png(
    pdf_path: Path,
    out_png: Path,
    *,
    dpi: int = 200,
) -> Path | None:
    """Render page 1 of ``pdf_path`` to ``out_png`` (RGB PNG). Returns ``out_png`` on success."""

    try:
        import pdf2image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Install pdf2image (see requirements.txt) and system Poppler for PDF rendering."
        ) from e

    if not pdf_path.is_file():
        return None

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.exists():
        try:
            Image.open(out_png).verify()
            return out_png
        except Exception:
            try:
                out_png.unlink()
            except OSError:
                pass

    try:
        pages = pdf2image.convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=1,
            last_page=1,
        )
    except Exception:
        return None

    if not pages:
        return None

    try:
        pages[0].convert("RGB").save(out_png, format="PNG")
    except OSError:
        return None
    return out_png
