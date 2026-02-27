#!/usr/bin/env python3
"""Streamlined 2D global conformal block and crossing demo.

This script is a compact, script-friendly counterpart of the 2D section
in the Mathematica notebook demos. It evaluates

  R(x) = v^Delta_phi - u^Delta_phi + sum_lambda lambda^2 * (v^Delta_phi G - u^Delta_phi G_cross)

along the diagonal z = zbar = x for a toy truncated spectrum.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mpmath as mp


def parse_args() -> argparse.Namespace:
    """CLI arguments for precision, diagonal range, and output path."""
    parser = argparse.ArgumentParser(description="2D conformal block crossing demo")
    parser.add_argument("--delta-phi", type=float, default=0.518, help="external scalar dimension")
    parser.add_argument("--x-min", type=float, default=0.05)
    parser.add_argument("--x-max", type=float, default=0.45)
    parser.add_argument("--x-step", type=float, default=0.01)
    parser.add_argument("--precision", type=int, default=60, help="decimal digits")
    parser.add_argument("--out-dir", type=Path, default=Path("streamlined_output_2d"))
    return parser.parse_args()


def k_beta(beta: mp.mpf, x: mp.mpf) -> mp.mpf:
    """Holomorphic building block k_beta(x)."""
    return x ** (beta / 2) * mp.hyp2f1(beta / 2, beta / 2, beta, x)


def global_block_2d(delta: mp.mpf, ell: int, z: mp.mpf, zb: mp.mpf) -> mp.mpf:
    """Symmetrized scalar global block used for simple residual checks."""
    return k_beta(delta + ell, z) * k_beta(delta - ell, zb) + k_beta(delta + ell, zb) * k_beta(delta - ell, z)


def crossing_vector_2d(delta_phi: mp.mpf, delta: mp.mpf, ell: int, z: mp.mpf, zb: mp.mpf) -> mp.mpf:
    """Single-operator contribution to the scalar crossing equation."""
    u = z * zb
    v = (1 - z) * (1 - zb)
    g = global_block_2d(delta, ell, z, zb)
    g_cross = global_block_2d(delta, ell, 1 - z, 1 - zb)
    return v ** delta_phi * g - u ** delta_phi * g_cross


def truncated_crossing_residual(delta_phi: mp.mpf, spectrum: list[tuple[float, int, float]], x: mp.mpf) -> mp.mpf:
    """Residual of crossing at one diagonal point x with a finite spectrum."""
    z = x
    zb = x
    u = z * zb
    v = (1 - z) * (1 - zb)

    # Identity contribution for identical external scalars.
    residual = v ** delta_phi - u ** delta_phi

    for delta, ell, ope_sq in spectrum:
        residual += mp.mpf(ope_sq) * crossing_vector_2d(delta_phi, mp.mpf(delta), ell, z, zb)

    return residual


def frange(start: float, end: float, step: float) -> list[float]:
    """Floating-point range helper including the right endpoint within tolerance."""
    values: list[float] = []
    x = start
    while x <= end + abs(step) * 1e-9:
        values.append(x)
        x += step
    return values


def main() -> int:
    args = parse_args()
    if args.x_step <= 0:
        raise ValueError("x-step must be positive")
    if args.x_min <= 0 or args.x_max >= 1:
        raise ValueError("x range should stay inside (0, 1)")

    mp.mp.dps = args.precision
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Small toy spectrum inspired by low-lying 3D Ising data (for demo only).
    spectrum = [
        (1.4126, 0, 0.24),
        (3.8303, 2, 0.016),
        (5.50, 4, 0.002),
    ]

    xs = frange(args.x_min, args.x_max, args.x_step)
    rows: list[dict[str, str]] = []

    print(f"Evaluating {len(xs)} points with dps={args.precision}...")
    for x in xs:
        residual = truncated_crossing_residual(mp.mpf(args.delta_phi), spectrum, mp.mpf(x))
        rows.append(
            {
                "x": f"{x:.6f}",
                "residual_real": mp.nstr(mp.re(residual), 30),
                "residual_abs": mp.nstr(abs(residual), 30),
            }
        )

    csv_path = args.out_dir / "crossing_residual_diagonal.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "residual_real", "residual_abs"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {csv_path}")
    print("Tip: residual_abs should become smaller with a better spectrum truncation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
