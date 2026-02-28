#!/usr/bin/env python3
"""Plot mixed-correlator 3D Ising island scan results.

Reads the scan_results.csv (and optionally scan_results.json) produced by
ising3d_mixed_island_scan.py and generates a scatter plot of the
(Delta_sigma, Delta_epsilon) grid, color-coded by feasibility:

  - Green circles: allowed points (inside the island).
  - Red crosses:   excluded points (outside the island).
  - Green shading: convex envelope of allowed points.

If scan_results.json is provided, the plot is annotated with the truncation
parameters (k_max, l_max, etc.) for reproducibility.

Usage
-----
  python3 plot_ising3d_mixed_island.py --csv results/scan_results.csv
  python3 plot_ising3d_mixed_island.py --csv results/scan_results.csv \\
      --json results/scan_results.json --out-prefix my_island

Output
------
  <out-prefix>.png  and  <out-prefix>.pdf
  Defaults to ising3d_mixed_island.{png,pdf} in the same directory as the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_bool(value: str) -> bool | None:
    """Parse CSV boolean field allowing empty values."""
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "":
        return None
    return None


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""
    parser = argparse.ArgumentParser(description="Plot 3D Ising mixed island scan")
    parser.add_argument("--csv", type=Path, required=True, help="scan_results.csv from mixed scan")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional scan_results.json for config annotation",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="output path prefix (without extension). Defaults beside CSV.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="3D Ising mixed-correlator island scan",
        help="figure title",
    )
    return parser.parse_args()


def main() -> int:
    """Read scan results and create PNG/PDF island plots."""
    args = parse_args()

    # Load the CSV produced by ising3d_mixed_island_scan.py.
    rows: list[dict[str, str]] = []
    with args.csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    # Partition into allowed / excluded / error points.
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    allowed_rows = [r for r in ok_rows if parse_bool(r.get("allowed", "")) is True]
    excluded_rows = [r for r in ok_rows if parse_bool(r.get("allowed", "")) is False]

    if not ok_rows:
        raise SystemExit("No successful points to plot (status=ok missing).")

    # Optionally load JSON metadata for annotation (truncation parameters, etc.).
    metadata = None
    if args.json is not None and args.json.is_file():
        with args.json.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = args.csv.parent / "ising3d_mixed_island"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required for plotting. "
            "Install it in your active environment and retry."
        ) from exc

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=160)

    if excluded_rows:
        x_ex = [float(r["delta_sigma"]) for r in excluded_rows]
        y_ex = [float(r["delta_epsilon"]) for r in excluded_rows]
        ax.scatter(
            x_ex,
            y_ex,
            s=20,
            marker="x",
            color="#c23b22",
            alpha=0.75,
            label=f"Excluded ({len(excluded_rows)})",
        )

    if allowed_rows:
        x_al = [float(r["delta_sigma"]) for r in allowed_rows]
        y_al = [float(r["delta_epsilon"]) for r in allowed_rows]
        ax.scatter(
            x_al,
            y_al,
            s=26,
            marker="o",
            facecolor="#1b8a5a",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            label=f"Allowed ({len(allowed_rows)})",
        )

        # Draw a vertical-slice envelope: for each sigma value, shade between
        # the min and max allowed epsilon. This approximates the island boundary.
        by_sigma: dict[float, list[float]] = {}
        for sx, ey in zip(x_al, y_al):
            by_sigma.setdefault(sx, []).append(ey)

        sigmas = sorted(by_sigma)
        if len(sigmas) >= 2:
            lower = [min(by_sigma[s]) for s in sigmas]
            upper = [max(by_sigma[s]) for s in sigmas]
            ax.fill_between(sigmas, lower, upper, color="#1b8a5a", alpha=0.12, label="Allowed envelope")

    ax.set_xlabel("Delta_sigma")
    ax.set_ylabel("Delta_epsilon (first Z2-even scalar)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)

    if metadata and isinstance(metadata, dict):
        cfg = metadata.get("config", {})
        if isinstance(cfg, dict):
            try:
                text = (
                    f"k={cfg.get('k_max')} l={cfg.get('l_max')} m={cfg.get('m_max')} n={cfg.get('n_max')}"
                    f" | cutoff={cfg.get('cutoff')} | dim={cfg.get('dim')}"
                )
                ax.text(
                    0.02,
                    0.98,
                    text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="#999999"),
                )
            except Exception:
                pass

    ax.legend(loc="best")
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
