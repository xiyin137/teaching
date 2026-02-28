#!/usr/bin/env python3
"""Combine fan-out Harvard mixed-island slice outputs into one dataset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sigma-slice mixed-island outputs into combined CSV/JSON."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base output directory containing sigma_* slice subdirectories.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output prefix for combined files (default: <base-dir>/scan_results_combined).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also generate combined island plot via plot_ising3d_mixed_island.py.",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python executable used to call the plotting script.",
    )
    return parser.parse_args()


def _parse_bool(value: str) -> bool | None:
    value = value.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def _read_slice_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {
                "delta_sigma": float(row["delta_sigma"]),
                "delta_epsilon": float(row["delta_epsilon"]),
                "allowed": _parse_bool(row.get("allowed", "")),
                "elapsed_sec": float(row.get("elapsed_sec", "nan")),
                "status": row.get("status", ""),
                "error": row.get("error", ""),
                "point_dir": row.get("point_dir", ""),
                "slice_csv": str(csv_path),
            }
            rows.append(parsed)
    return rows


def _write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "delta_sigma",
        "delta_epsilon",
        "allowed",
        "elapsed_sec",
        "status",
        "error",
        "point_dir",
        "slice_csv",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    out_prefix = args.out_prefix.resolve() if args.out_prefix else (base_dir / "scan_results_combined")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    slice_csv_paths = sorted(base_dir.glob("sigma_*/scan_results.csv"))
    if not slice_csv_paths:
        raise SystemExit(f"No slice CSV files found under: {base_dir}")

    rows: list[dict[str, Any]] = []
    for csv_path in slice_csv_paths:
        rows.extend(_read_slice_csv(csv_path))

    rows.sort(key=lambda r: (r["delta_sigma"], r["delta_epsilon"], r["point_dir"]))

    csv_out = out_prefix.with_suffix(".csv")
    json_out = out_prefix.with_suffix(".json")
    _write_combined_csv(csv_out, rows)

    allowed_count = sum(1 for r in rows if r["status"] == "ok" and r["allowed"] is True)
    excluded_count = sum(1 for r in rows if r["status"] == "ok" and r["allowed"] is False)
    error_count = sum(1 for r in rows if r["status"] != "ok")

    payload = {
        "base_dir": str(base_dir),
        "slice_csv_files": [str(p) for p in slice_csv_paths],
        "summary": {
            "total_points": len(rows),
            "allowed_points": allowed_count,
            "excluded_points": excluded_count,
            "error_points": error_count,
        },
        "results": rows,
    }
    with json_out.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote combined CSV: {csv_out}")
    print(f"Wrote combined JSON: {json_out}")
    print(
        "Summary: "
        f"allowed={allowed_count}, excluded={excluded_count}, errors={error_count}, total={len(rows)}"
    )

    if args.plot:
        plot_script = Path(__file__).with_name("plot_ising3d_mixed_island.py")
        plot_prefix = out_prefix.with_name(f"{out_prefix.name}_plot")
        cmd = [
            args.python_bin,
            str(plot_script),
            "--csv",
            str(csv_out),
            "--json",
            str(json_out),
            "--out-prefix",
            str(plot_prefix),
        ]
        print("Running plot command:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: plot generation failed ({exc}).", file=sys.stderr)


if __name__ == "__main__":
    main()
