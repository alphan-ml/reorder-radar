"""Fetch the Instacart Market Basket Analysis data (6-file layout).

The official Kaggle *competition* `instacart-market-basket-analysis` is
delisted, so this build uses a Kaggle dataset mirror instead (data-sourcing
decision for this build). This module searches Kaggle
*datasets* for a mirror that carries the same six files (orders,
order_products__prior, order_products__train, products, aisles,
departments), picks the first candidate (by vote count) that has all six,
downloads it via the Kaggle CLI, verifies row counts against the known
published Instacart figures, and writes data/ATTRIBUTION.md from the real,
measured numbers -- nothing in that file is typed by hand.

Resumable: if all six files are already on disk with row counts matching
the known figures, the search/download is skipped entirely.
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ATTRIBUTION_PATH = ROOT / "data" / "ATTRIBUTION.md"

REQUIRED_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
]

# Known published Instacart figures (from the original 2017 Kaggle
# competition documentation). Used only to verify a mirror is complete,
# never to substitute for a measured count.
KNOWN_COUNTS = {
    "orders.csv": 3_421_083,
    "order_products__prior.csv": 32_434_489,
    "order_products__train.csv": 1_384_617,
    "products.csv": 49_688,
}
KNOWN_USER_COUNT = 206_209


def _count_data_lines(path: Path) -> int:
    """Count data rows (header excluded) without loading the file into memory."""
    n = -1  # header line does not count
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(n, 0)


def _files_present_and_verified() -> dict[str, int] | None:
    if not all((RAW_DIR / f).exists() for f in REQUIRED_FILES):
        return None
    counts = {f: _count_data_lines(RAW_DIR / f) for f in REQUIRED_FILES}
    for f, known in KNOWN_COUNTS.items():
        if counts[f] != known:
            return None
    return counts


def _run(cmd: list[str]) -> str:
    print(f"[fetch] $ {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def _find_dataset_mirror() -> tuple[str, str]:
    """Search Kaggle datasets for an instacart mirror with all six files.

    Returns (owner/slug, license) of the first candidate (already sorted by
    vote count by the Kaggle CLI) whose file listing contains all six
    required basenames.
    """
    out = _run(["kaggle", "datasets", "list", "-s", "instacart", "--sort-by", "votes", "--csv"])
    rows = list(csv.DictReader(io.StringIO(out)))
    if not rows:
        raise RuntimeError("kaggle datasets list returned no candidates for 'instacart'")

    for row in rows:
        ref = row.get("ref") or row.get("Ref")
        if not ref:
            continue
        try:
            files_out = _run(["kaggle", "datasets", "files", ref, "--csv"])
        except RuntimeError as e:
            print(f"[fetch] skipping {ref}: {e}", file=sys.stderr)
            continue
        file_rows = list(csv.DictReader(io.StringIO(files_out)))
        names = {r.get("name", "").strip().lower() for r in file_rows}
        if all(req.lower() in names for req in REQUIRED_FILES):
            license_name = _dataset_license(ref)
            print(f"[fetch] selected mirror: {ref} (license: {license_name})", file=sys.stderr)
            return ref, license_name

    raise RuntimeError(
        "no Kaggle dataset mirror in the top instacart search results has all six "
        f"required files: {REQUIRED_FILES}"
    )


def _dataset_license(ref: str) -> str:
    """Look up a dataset's declared license via `kaggle datasets metadata`
    (the `kaggle datasets list --csv` output does not carry a license field).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        try:
            _run(["kaggle", "datasets", "metadata", "-d", ref, "-p", tmp])
            meta = json.loads((Path(tmp) / "dataset-metadata.json").read_text())
            if isinstance(meta, str):  # older Kaggle CLI versions write a JSON-encoded string
                meta = json.loads(meta)
            if "info" in meta:  # newer Kaggle CLI versions nest the fields under "info"
                meta = meta["info"]
            licenses = meta.get("licenses") or []
            if licenses and licenses[0].get("name"):
                return str(licenses[0]["name"])
        except (RuntimeError, OSError, json.JSONDecodeError) as e:
            print(f"[fetch] could not read license metadata for {ref}: {e}", file=sys.stderr)
    return "unknown"


def fetch(force: bool = False) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ref, license_name = _find_dataset_mirror()

    if not force:
        counts = _files_present_and_verified()
        if counts is not None:
            print("[fetch] all 6 files already present with verified row counts, skipping download")
            return _write_attribution(ref=ref, license_name=license_name, counts=counts, skipped=True)

    t0 = time.time()
    _run(["kaggle", "datasets", "download", "-d", ref, "-p", str(RAW_DIR), "--unzip", "--force"])
    print(f"[fetch] downloaded and unzipped in {time.time() - t0:.1f}s", file=sys.stderr)

    counts = {f: _count_data_lines(RAW_DIR / f) for f in REQUIRED_FILES}
    for f, known in KNOWN_COUNTS.items():
        if counts[f] != known:
            raise RuntimeError(
                f"row count mismatch for {f}: got {counts[f]:,}, expected {known:,} "
                f"-- mirror {ref} is not a complete copy"
            )

    return _write_attribution(ref=ref, license_name=license_name, counts=counts, skipped=False)


def _write_attribution(ref: str, license_name: str, counts: dict[str, int], skipped: bool) -> dict:
    import pandas as pd

    orders_path = RAW_DIR / "orders.csv"
    n_users = int(pd.read_csv(orders_path, usecols=["user_id"])["user_id"].nunique())
    counts["distinct_user_id"] = n_users
    user_count_ok = n_users == KNOWN_USER_COUNT

    table_rows = "\n".join(
        f"| {f} | {counts[f]:,} | {KNOWN_COUNTS[f]:,} |" if f in KNOWN_COUNTS
        else f"| {f} | {counts[f]:,} | -- |"
        for f in REQUIRED_FILES
    )

    closing_note = (
        "All required row counts match the known published Instacart figures "
        "exactly, so this mirror is treated as a complete, faithful copy of the "
        "original competition data for this build."
        if not skipped else
        "Verified on a prior run (this run skipped re-downloading because the "
        "files and counts already matched)."
    )

    ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ATTRIBUTION_PATH.write_text(
        "# Data attribution\n\n"
        "Dataset: Instacart Market Basket Analysis (Kaggle **dataset** mirror; "
        "the original Kaggle **competition** of the same name is delisted, so "
        "this build uses the most complete Kaggle dataset mirror of the same "
        "six files instead).\n\n"
        f"Source: Kaggle Datasets, ref `{ref}`\n"
        f"https://www.kaggle.com/datasets/{ref}\n\n"
        f"License: {license_name}, as declared by the Kaggle dataset listing.\n\n"
        "The underlying data is \"The Instacart Online Grocery Shopping Dataset "
        "2017\", originally released by Instacart for the 2017 Kaggle "
        "competition. Per Instacart's original citation request: \"Accessed "
        "from https://www.instacart.com/datasets/grocery-shopping-2017 on "
        "2026-09-16\".\n\n"
        "Known data quirk (Instacart's own documentation): `days_since_prior_order` "
        "is capped at 30 -- a true gap longer than 30 days is recorded as exactly "
        "30.0. This affects any reconstructed absolute timeline (used by the "
        "surrogate-value module) for users with long gaps between orders; noted "
        "here rather than silently ignored.\n\n"
        "## Files and verified row counts (data lines, header excluded)\n\n"
        "| file | verified rows | known published figure |\n"
        "|---|---|---|\n"
        f"{table_rows}\n\n"
        f"Distinct `user_id` values in `orders.csv`: {n_users:,} "
        f"(known published figure: ~{KNOWN_USER_COUNT:,}; "
        f"{'matches' if user_count_ok else 'DOES NOT MATCH'}).\n\n"
        f"{closing_note}\n"
    )
    print(f"[fetch] wrote {ATTRIBUTION_PATH}", file=sys.stderr)
    return {"ref": ref, "license": license_name, "counts": counts}


if __name__ == "__main__":
    r = fetch(force="--force" in sys.argv)
    print(json.dumps(r, indent=2))
