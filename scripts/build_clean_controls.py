"""Build clean-control sets per family (spec §4.6).

Pilot defaults (**250** TRN / **100** TST per family) match the specification;
use ``--scale full`` with optional ``--full-multiplier`` (**5–10**, default **7**)
for the scaled corpus.

RCT (receipt rasters):

    python -m scripts.build_clean_controls --family rct --pool TRN --scale pilot
    python -m scripts.build_clean_controls --family rct --pool TST --n 100 --source sroie
    python -m scripts.build_clean_controls --family rct --pool TRN --source cord
    python -m scripts.build_clean_controls --family rct --pool TRN --source findit

EML (rendered emails):

    python -m scripts.build_clean_controls --family eml --pool TRN --scale pilot
    python -m scripts.build_clean_controls --family eml --pool TST --eml-source avocado_mail
    python -m scripts.build_clean_controls --family eml --pool TRN --eml-source rvl_cdip_email

DOC (document rasters — requires phase-0 loader):

    python -m scripts.build_clean_controls --family doc --pool TRN --scale pilot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sec.clean_controls import (  # noqa: E402
    build_clean_controls,
    build_doc_clean_controls,
    build_eml_clean_controls,
    default_clean_count,
)
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import PoolSplit  # noqa: E402
from scripts.phase0_setup import build_loader as build_sroie_loader  # noqa: E402


LOG = new_logger("sec.build_clean_controls")


def _load_split(cfg, source: str) -> PoolSplit:
    name_map = {
        "sroie": "pool_split_sroie.yaml",
        "cord": "pool_split_cord.yaml",
        "findit": "pool_split_findit.yaml",
        "enron_mail": "pool_split_enron.yaml",
        "avocado_mail": "pool_split_avocado_mail.yaml",
        "rvl_cdip_email": "pool_split_rvl_cdip_email.yaml",
        "doc": "pool_split_doc.yaml",
    }
    filename = name_map.get(source, f"pool_split_{source}.yaml")
    sidecar = cfg.project_root / "configs" / filename
    if not sidecar.exists():
        raise SystemExit(
            f"Missing {sidecar}. Run the appropriate phase-0 setup first "
            f"(`scripts.phase0_setup` for sroie, `scripts.phase0_setup_cord` for cord, "
            f"`scripts.phase0_setup_findit2` for findit, `scripts.phase0_setup_mail` for mail splits, "
            f"`scripts.phase0_setup_rvl_cdip_email` for RVL email)."
        )
    with open(sidecar, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PoolSplit(
        train_ids=tuple(data["train_ids"]),
        test_ids=tuple(data["test_ids"]),
    )


def _build_rct_loader(cfg, source: str):
    if source == "sroie":
        return build_sroie_loader(cfg)
    if source == "cord":
        from scripts.phase0_setup_cord import build_cord_loader

        return build_cord_loader(cfg)
    if source == "findit":
        from scripts.phase0_setup_findit2 import build_findit2_loader

        return build_findit2_loader(cfg)
    raise SystemExit(f"Unknown RCT source {source!r}; use 'sroie', 'cord', or 'findit'")


def _build_eml_loader(cfg, eml_source: str):
    if eml_source == "enron_mail":
        from sec.sources.enron_mail import build_enron_mail_loader

        return build_enron_mail_loader(cfg.project_root, cfg.source("enron"))
    if eml_source == "avocado_mail":
        from sec.sources.avocado_mail import AvocadoMailLoader

        src = cfg.source("avocado")
        if src.root is None:
            raise SystemExit("Configure sources.avocado.root in configs/paths.yaml")
        return AvocadoMailLoader(src.root)
    if eml_source == "rvl_cdip_email":
        from sec.sources.rvl_cdip_email import build_rvl_cdip_email_loader

        try:
            return build_rvl_cdip_email_loader(cfg.source("rvl_cdip_email"))
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
    raise SystemExit(f"Unknown EML source {eml_source!r}")


def _default_eml_source(pool: str) -> str:
    return "enron_mail" if pool.upper() == "TRN" else "avocado_mail"


def _eml_batch_suffix(eml_source: str) -> str:
    if eml_source == "rvl_cdip_email":
        return "RVLCDIP"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SEC clean-control artifacts")
    parser.add_argument("--family", choices=["rct", "eml", "doc"], default="rct")
    parser.add_argument("--pool", choices=["TRN", "TST"], required=True)
    parser.add_argument(
        "--scale",
        choices=["pilot", "full"],
        default="pilot",
        help="pilot = 250 TRN / 100 TST per family; full = multiply pilot baseline",
    )
    parser.add_argument(
        "--full-multiplier",
        type=float,
        default=None,
        help="Scale factor when --scale full (spec: 5–10× pilot); default 7",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override target count (otherwise derived from --scale)",
    )
    parser.add_argument(
        "--source",
        choices=["sroie", "cord", "findit"],
        default="sroie",
        help="RCT raster source dataset (EML/DOC ignore this)",
    )
    parser.add_argument(
        "--eml-source",
        choices=["enron_mail", "avocado_mail", "rvl_cdip_email"],
        default=None,
        help="Mail loader for --family eml (defaults: TRN→enron, TST→avocado)",
    )
    parser.add_argument("--round-trip-fraction", type=float, default=0.1)
    args = parser.parse_args()

    n = (
        args.n
        if args.n is not None
        else default_clean_count(args.pool, scale=args.scale, full_multiplier=args.full_multiplier)
    )

    configure_root_logger()
    cfg = load_config()

    if args.family == "rct":
        loader = _build_rct_loader(cfg, args.source)
        split = _load_split(cfg, args.source)
        if args.source == "sroie":
            suffix = ""
        elif args.source == "findit":
            suffix = "FIN"
        else:
            suffix = args.source.upper()
        counts = build_clean_controls(
            cfg,
            pool=args.pool,
            count=n,
            loader=loader,
            split=split,
            round_trip_fraction=args.round_trip_fraction,
            batch_suffix=suffix,
        )
        LOG.info(
            "RCT clean (%s) %s: written=%d round_tripped=%d skipped=%d",
            args.source,
            args.pool,
            counts.written,
            counts.round_tripped,
            counts.skipped,
        )
        return 0

    if args.family == "eml":
        eml_src = args.eml_source or _default_eml_source(args.pool)
        if eml_src == "rvl_cdip_email" and args.pool.upper() != "TRN":
            raise SystemExit("RVL-CDIP EML clean controls are defined for TRN only.")
        loader = _build_eml_loader(cfg, eml_src)
        split_key = eml_src if eml_src != "rvl_cdip_email" else "rvl_cdip_email"
        split = _load_split(cfg, split_key)
        suffix = _eml_batch_suffix(eml_src)
        counts = build_eml_clean_controls(
            cfg,
            pool=args.pool,
            count=n,
            loader=loader,
            split=split,
            round_trip_fraction=args.round_trip_fraction,
            batch_suffix=suffix,
        )
        LOG.info(
            "EML clean (%s) %s: written=%d round_tripped=%d skipped=%d",
            eml_src,
            args.pool,
            counts.written,
            counts.round_tripped,
            counts.skipped,
        )
        return 0

    from scripts.phase0_setup_doc import build_doc_loader  # noqa: E402

    loader = build_doc_loader(cfg)
    split = _load_split(cfg, "doc")
    counts = build_doc_clean_controls(
        cfg,
        pool=args.pool,
        count=n,
        loader=loader,
        split=split,
        round_trip_fraction=args.round_trip_fraction,
    )
    LOG.info(
        "DOC clean %s: written=%d round_tripped=%d skipped=%d",
        args.pool,
        counts.written,
        counts.round_tripped,
        counts.skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
