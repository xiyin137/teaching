#!/usr/bin/env python3
"""Conformal blocks and bootstrap demo (production-ready streamlined rewrite).

Background
----------
In a conformal field theory (CFT), the operator product expansion (OPE)
decomposes four-point correlators into conformal blocks -- kinematically
determined functions labeled by the exchanged operator's dimension Delta
and spin ell.  Crossing symmetry (associativity of the OPE) yields
nontrivial constraints that form the basis of the conformal bootstrap.

This script mirrors the main sections of the companion Mathematica notebook
``Conformal blocks and bootstrap demo.nb``, using mpmath for arbitrary-
precision arithmetic and PyCFTBoot+SDPB for the 3D numerical bootstrap.

Sections
--------
1) **2D global blocks** -- In 2D the global conformal group is SL(2,R) x SL(2,R),
   so blocks factorize into holomorphic pieces k_beta(z) * k_beta(zbar).
   We compute crossing residuals, P2D derivative coefficients, and gap scans.

2) **4D scalar blocks** -- Blocks take the Dolan-Osborn form with a 1/(z-zbar)
   prefactor and satisfy the conformal Casimir equation.  We verify the
   Casimir eigenvalue numerically and compute crossing-derivative vectors.

3) **3D Ising bootstrap** -- Conformal blocks are computed numerically via
   PyCFTBoot's Zamolodchikov-like recursion, then fed to SDPB to derive
   a rigorous upper bound on the leading Z2-even scalar dimension.

Usage
-----
  python3 conformal_blocks_and_bootstrap_demo_streamlined.py 2d
  python3 conformal_blocks_and_bootstrap_demo_streamlined.py 4d
  python3 conformal_blocks_and_bootstrap_demo_streamlined.py 3d --delta-sigma 0.518

Parameter glossary
------------------
  Delta, delta      conformal dimension of an exchanged operator
  spin, ell         spin of an exchanged operator
  z, zbar           cross-ratio variables
  u, v              u = z*zbar, v = (1-z)*(1-zbar)
  delta_phi         external scalar dimension (identical scalars)
  delta12, delta34  external dimension differences (Delta1-Delta2, Delta3-Delta4)
  OPE^2             squared OPE coefficient

References
----------
  - Dolan, Osborn, hep-th/0309180 (4D conformal blocks)
  - Rattazzi, Rychkov, Tonni, Vichi (RRTV), 0807.0004 (bootstrap bounds)
  - Kos, Poland, Simmons-Duffin, 1406.4858 (PyCFTBoot / 3D blocks)
  - Simmons-Duffin, 1502.02033 (SDPB solver, bootstrap review)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mpmath as mp

LOCAL_CUTOFF_STABLE_MIN = 0.15
LOCAL_CUTOFF_STABLE_MAX = 0.20


# ===================== 2D section =====================
# In 2D the global conformal group is SL(2,R) x SL(2,R).  Conformal blocks
# factorize: G_{Delta,ell}(z,zbar) = k_{Delta+ell}(z) k_{Delta-ell}(zbar) + (z<->zbar).
# The crossing equation for identical scalars phi with dimension delta_phi is:
#   v^{delta_phi} G(z,zbar) = u^{delta_phi} G(1-z,1-zbar),
# where u = z*zbar, v = (1-z)*(1-zbar).


def k_beta(beta: mp.mpf, x: mp.mpf) -> mp.mpf:
    """SL(2,R) conformal block (holomorphic building block).

    k_beta(x) = x^{beta/2} _2F_1(beta/2, beta/2; beta; x).

    This is the eigenfunction of the SL(2,R) Casimir operator and the
    fundamental building block for 2D global conformal blocks.
    """
    return x ** (beta / 2) * mp.hyp2f1(beta / 2, beta / 2, beta, x)


def block2d(delta: mp.mpf, spin: int, z: mp.mpf, zb: mp.mpf, delta12: mp.mpf, delta34: mp.mpf) -> mp.mpf:
    """2D global conformal block for general external dimensions.

    G_{Delta,ell}(z,zbar) = z^{h} zbar^{hbar} _2F_1(a1,a2;2h;z) _2F_1(b1,b2;2hbar;zbar)

    where h = (Delta+ell)/2, hbar = (Delta-ell)/2, and a1,a2,b1,b2 depend on
    the external dimension differences delta12 = Delta1-Delta2, delta34 = Delta3-Delta4.
    For identical external scalars, set delta12 = delta34 = 0.
    """
    a1 = (delta + spin - delta12) / 2
    a2 = (delta + spin + delta34) / 2
    b1 = (delta - spin - delta12) / 2
    b2 = (delta - spin + delta34) / 2
    return (
        z ** ((delta + spin) / 2)
        * zb ** ((delta - spin) / 2)
        * mp.hyp2f1(a1, a2, delta + spin, z)
        * mp.hyp2f1(b1, b2, delta - spin, zb)
    )


def crossing_block_2d(delta_phi: mp.mpf, delta: mp.mpf, spin: int, z: mp.mpf, zb: mp.mpf) -> mp.mpf:
    """Crossing vector for one operator in the OPE (identical external scalars).

    F_{Delta,ell}(z,zbar) = v^{delta_phi} G(z,zbar) - u^{delta_phi} G(1-z,1-zbar).

    The full crossing equation is: identity + sum_i OPE_i^2 * F_i = 0.
    """
    b = block2d(delta, spin, z, zb, mp.mpf("0"), mp.mpf("0"))
    bx = block2d(delta, spin, 1 - z, 1 - zb, mp.mpf("0"), mp.mpf("0"))
    return ((1 - z) * (1 - zb)) ** delta_phi * b - (z * zb) ** delta_phi * bx


def p2d_coeffs(delta_phi: mp.mpf, delta: mp.mpf, spin: int) -> tuple[mp.mpf, mp.mpf]:
    """P2D crossing coefficients: x^1 and x^3 Taylor coefficients.

    Expands the crossing block on the diagonal z = zbar = 1/2 + x around
    the crossing-symmetric point x = 0.  These two components form the
    crossing-derivative vector used in the simplest RRTV (0807.0004)
    bootstrap bound.
    """

    def fx(x: mp.mpf) -> mp.mpf:
        z = mp.mpf("0.5") + x
        return crossing_block_2d(delta_phi, delta, spin, z, z)

    c1 = mp.diff(fx, mp.mpf("0"), 1)
    c3 = mp.diff(fx, mp.mpf("0"), 3) / mp.mpf(math.factorial(3))
    return c1, c3


def frange(start: float, end: float, step: float) -> list[float]:
    """Inclusive floating-point range helper with small endpoint tolerance."""
    vals: list[float] = []
    x = start
    while x <= end + abs(step) * 1e-9:
        vals.append(x)
        x += step
    return vals


def run_2d(args: argparse.Namespace) -> int:
    """Run 2D outputs: diagonal residuals, P2D table, and gap-style scans.

    Outputs:
      - crossing_2d_diagonal.csv: crossing residual on z=zbar diagonal for a toy spectrum
      - p2d_spin0_table.csv: P2D coefficients vs operator dimension (spin 0)
      - p2d_gap_scan_gap{0,1,2}.csv: gap scans for spins 0,2,4
    """
    mp.mp.dps = args.precision
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_phi = mp.mpf(args.delta_phi)

    # Toy truncated spectrum: (Delta, spin, OPE^2).
    # Approximate 3D Ising values used as a pedagogical stand-in.
    # With only 3 operators the crossing residual is small but nonzero;
    # a complete OPE would give exactly zero.
    demo_spectrum = [
        (mp.mpf("1.4126"), 0, mp.mpf("0.24")),
        (mp.mpf("3.8303"), 2, mp.mpf("0.016")),
        (mp.mpf("5.50"), 4, mp.mpf("0.002")),
    ]

    rows: list[dict[str, str]] = []
    for x in frange(args.x_min, args.x_max, args.x_step):
        z = mp.mpf(x)
        residual = (1 - z) ** (2 * delta_phi) - z ** (2 * delta_phi)
        for delta, spin, ope2 in demo_spectrum:
            residual += ope2 * crossing_block_2d(delta_phi, delta, spin, z, z)
        rows.append(
            {
                "x": f"{x:.6f}",
                "residual_real": mp.nstr(mp.re(residual), 30),
                "residual_abs": mp.nstr(abs(residual), 30),
            }
        )

    diag_csv = out_dir / "crossing_2d_diagonal.csv"
    with diag_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "residual_real", "residual_abs"])
        writer.writeheader()
        writer.writerows(rows)

    # P2D table: x^1 and x^3 crossing coefficients as a function of operator
    # dimension (spin 0).  Reproduces the notebook's Table 1 for cross-checking.
    table_csv = out_dir / "p2d_spin0_table.csv"
    with table_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["weight", "p2d_x", "p2d_x3"])
        writer.writeheader()
        for weight in frange(args.weight_min, args.weight_max, args.weight_step):
            c1, c3 = p2d_coeffs(delta_phi, mp.mpf(weight), spin=0)
            writer.writerow(
                {
                    "weight": f"{weight:.6f}",
                    "p2d_x": mp.nstr(c1, 30),
                    "p2d_x3": mp.nstr(c3, 30),
                }
            )

    # Gap-style scans: sweep operator dimensions for spins 0,2,4, starting
    # from the assumed gap (or unitarity bound).  The 1.1^spin * 2^weight
    # rescaling improves visibility across orders of magnitude.
    gaps = [0.0, 1.0, 2.0]
    for gap in gaps:
        gap_csv = out_dir / f"p2d_gap_scan_gap{int(gap)}.csv"
        with gap_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["spin", "weight", "p2d_x_scaled", "p2d_x3_scaled"],
            )
            writer.writeheader()
            for spin in (0, 2, 4):
                low = max(float(spin), gap if spin == 0 else 0.0)
                for weight in frange(low, args.gap_scan_weight_max, args.gap_scan_step):
                    c1, c3 = p2d_coeffs(delta_phi, mp.mpf(weight), spin=spin)
                    scale = (mp.mpf("1.1") ** spin) * (mp.mpf("2") ** mp.mpf(weight))
                    writer.writerow(
                        {
                            "spin": str(spin),
                            "weight": f"{weight:.6f}",
                            "p2d_x_scaled": mp.nstr(c1 / scale, 30),
                            "p2d_x3_scaled": mp.nstr(c3 / scale, 30),
                        }
                    )

    summary = {
        "diagonal_residual_csv": str(diag_csv),
        "p2d_table_csv": str(table_csv),
        "gap_scan_csvs": [str(out_dir / f"p2d_gap_scan_gap{i}.csv") for i in (0, 1, 2)],
    }
    with (out_dir / "summary_2d.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"2D demo complete: {out_dir}")
    return 0


# ===================== 4D section =====================
# In 4D, conformal blocks take the Dolan-Osborn form (hep-th/0309180)
# with a characteristic 1/(z-zbar) prefactor.  They satisfy the conformal
# Casimir equation: DD[G] = (1/2)(Delta(Delta-4) + ell(ell+2)) G,
# where DD is a second-order differential operator in (z, zbar).


def block4d(delta: mp.mpf, spin: int, z: mp.mpf, zb: mp.mpf, delta12: mp.mpf, delta34: mp.mpf) -> mp.mpf:
    """4D scalar conformal block (Dolan-Osborn, hep-th/0309180 Eq. 2.20).

    G_{Delta,ell}(z,zbar) = [z^{h+1} zbar^{hbar} F(.) F(.)
                            - (z <-> zbar)] / (z - zbar),

    where h = (Delta+ell)/2, hbar = (Delta-ell)/2.
    The 1/(z-zbar) pole is characteristic of even-dimensional blocks.
    A small shift is applied when z = zbar to avoid the singularity.
    """
    if z == zb:
        z = z + mp.mpf("1e-12")

    a1 = (delta + spin - delta12) / 2
    a2 = (delta + spin + delta34) / 2
    b1 = (delta - spin - delta12) / 2 - 1
    b2 = (delta - spin + delta34) / 2 - 1

    term1 = (
        z ** ((delta + spin) / 2 + 1)
        * zb ** ((delta - spin) / 2)
        / (z - zb)
        * mp.hyp2f1(a1, a2, delta + spin, z)
        * mp.hyp2f1(b1, b2, delta - spin - 2, zb)
    )
    term2 = (
        zb ** ((delta + spin) / 2 + 1)
        * z ** ((delta - spin) / 2)
        / (zb - z)
        * mp.hyp2f1(a1, a2, delta + spin, zb)
        * mp.hyp2f1(b1, b2, delta - spin - 2, z)
    )
    return term1 + term2


def dd_operator_4d(
    f,
    z: mp.mpf,
    zb: mp.mpf,
    d1: mp.mpf,
    d2: mp.mpf,
    d3: mp.mpf,
    d4: mp.mpf,
) -> mp.mpf:
    """4D conformal Casimir differential operator DD.

    For external dimensions d1..d4, applies the second-order operator
    DD to f(z, zbar) numerically via finite differences (mpmath.diff).
    The conformal block satisfies DD[G] = eigenvalue * G with
    eigenvalue = (1/2)(Delta(Delta-4) + ell(ell+2)).
    """
    dz = mp.diff(lambda t: f(t, zb), z, 1)
    dzz = mp.diff(lambda t: f(t, zb), z, 2)
    dzb = mp.diff(lambda t: f(z, t), zb, 1)
    dzbzb = mp.diff(lambda t: f(z, t), zb, 2)

    term_z = (
        z**2 * (1 - z) * dzz
        + (((d1 - d2 - d3 + d4) / 2) - 1) * z**2 * dz
        + ((d1 - d2) * (d3 - d4) / 4) * z * f(z, zb)
        + 2 * z * zb / (z - zb) * (1 - z) * dz
    )
    term_zb = (
        zb**2 * (1 - zb) * dzbzb
        + (((d1 - d2 - d3 + d4) / 2) - 1) * zb**2 * dzb
        + ((d1 - d2) * (d3 - d4) / 4) * zb * f(z, zb)
        + 2 * z * zb / (zb - z) * (1 - zb) * dzb
    )
    return term_z + term_zb


def crossing_block_4d(delta_phi: mp.mpf, delta: mp.mpf, spin: int, z: mp.mpf, zb: mp.mpf) -> mp.mpf:
    """4D crossing block with the (z-zbar) prefactor that cancels the pole.

    F = (z-zbar) [v^{delta_phi} G(z,zbar) - u^{delta_phi} G(1-z,1-zbar)].

    This combination is regular at z = zbar and suitable for Taylor expansion.
    """
    b = block4d(delta, spin, z, zb, mp.mpf("0"), mp.mpf("0"))
    bx = block4d(delta, spin, 1 - z, 1 - zb, mp.mpf("0"), mp.mpf("0"))
    return (z - zb) * (((1 - z) * (1 - zb)) ** delta_phi * b - (z * zb) ** delta_phi * bx)


def mixed_derivative_at_origin(func, m: int, n: int) -> mp.mpf:
    """Compute d^{m+n}f/dx^m dy^n at (0,0) by nested automatic differentiation."""
    return mp.diff(lambda x: mp.diff(lambda y: func(x, y), mp.mpf("0"), n), mp.mpf("0"), m)


def crossing_derivative_vector_4d(delta_phi: mp.mpf, delta: mp.mpf, spin: int, deg: int) -> list[tuple[int, int, mp.mpf]]:
    """Crossing-derivative vector F_{m,n} for the 4D bootstrap.

    Expands the crossing block around z = zbar = 1/2 and extracts the Taylor
    coefficients (1/m!n!) d^{m+n}F / dx^m dy^n.  Only the bootstrap-relevant
    components (n > m, n-m even) are kept; these encode the odd-parity
    derivatives under z <-> zbar exchange.
    """
    def local(x: mp.mpf, y: mp.mpf) -> mp.mpf:
        z = mp.mpf("0.5") + x
        zb = mp.mpf("0.5") + y
        return crossing_block_4d(delta_phi, delta, spin, z, zb)

    out: list[tuple[int, int, mp.mpf]] = []
    for m in range(0, 2 * deg + 1):
        for n in range(m + 2, 2 * deg - m + 1, 2):
            deriv = mixed_derivative_at_origin(local, m, n)
            coeff = deriv / mp.mpf(math.factorial(m) * math.factorial(n))
            out.append((m, n, coeff))
    return out


def run_4d(args: argparse.Namespace) -> int:
    """Run 4D outputs: Casimir residual checks, profile samples, derivative vectors.

    Outputs:
      - casimir_checks_4d.json: DD[G] - eigenvalue*G at test points (should be ~0)
      - block4d_diagonal_profile.csv: block values along the near-diagonal
      - crossing_derivatives_4d.csv: F_{m,n} vectors for sample operators
    """
    mp.mp.dps = args.precision
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify the Casimir equation DD[G] = eigenvalue * G at two generic complex
    # points.  Using complex z, zbar avoids accidental cancellations.
    # The eigenvalue is (1/2)(Delta(Delta-4) + ell(ell+2)) for d=4.
    casimir_points = [
        (mp.mpf("3.0"), 0, mp.mpf("0.4") + mp.mpf("0.5") * 1j, mp.mpf("0.5") - mp.mpf("0.3") * 1j),
        (mp.mpf("5.0"), 2, mp.mpf("0.3") + mp.mpf("0.2") * 1j, mp.mpf("0.45") - mp.mpf("0.15") * 1j),
    ]

    checks: list[dict[str, str]] = []
    for delta, spin, z, zb in casimir_points:
        f = lambda zz, zbb: block4d(delta, spin, zz, zbb, mp.mpf("0"), mp.mpf("0"))
        lhs = dd_operator_4d(f, z, zb, mp.mpf("2"), mp.mpf("2"), mp.mpf("2"), mp.mpf("2"))
        eig = mp.mpf("0.5") * (delta * (delta - 4) + spin * (spin + 2))
        rhs = eig * f(z, zb)
        checks.append(
            {
                "delta": mp.nstr(delta, 20),
                "spin": str(spin),
                "z": mp.nstr(z, 20),
                "zbar": mp.nstr(zb, 20),
                "casimir_residual_abs": mp.nstr(abs(lhs - rhs), 40),
            }
        )

    with (out_dir / "casimir_checks_4d.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)

    # Numeric profile of Block4D along the near-diagonal (z = x + i*eps, zbar = x - i*eps).
    # The small imaginary offset regularizes the 1/(z-zbar) pole.
    profile_csv = out_dir / "block4d_diagonal_profile.csv"
    with profile_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "block_real"])
        writer.writeheader()
        for x in frange(args.x_min, args.x_max, args.x_step):
            z = mp.mpf(x) + mp.mpf(args.diag_eps) * 1j
            zb = mp.mpf(x) - mp.mpf(args.diag_eps) * 1j
            val = block4d(mp.mpf("1.5"), 0, z, zb, mp.mpf("0"), mp.mpf("0"))
            writer.writerow({"x": f"{x:.6f}", "block_real": mp.nstr(mp.re(val), 30)})

    # Crossing-derivative vectors F_{m,n} for sample operators.
    # These are the components that enter the bootstrap SDP.
    vectors_csv = out_dir / "crossing_derivatives_4d.csv"
    with vectors_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delta", "spin", "m", "n", "coeff_real", "coeff_imag"])
        writer.writeheader()
        for delta, spin in ((3.0, 0), (5.0, 2)):
            vec = crossing_derivative_vector_4d(mp.mpf(args.delta_phi), mp.mpf(delta), spin, args.derivative_degree)
            for m, n, coeff in vec:
                writer.writerow(
                    {
                        "delta": f"{delta:.6f}",
                        "spin": str(spin),
                        "m": str(m),
                        "n": str(n),
                        "coeff_real": mp.nstr(mp.re(coeff), 30),
                        "coeff_imag": mp.nstr(mp.im(coeff), 30),
                    }
                )

    summary = {
        "casimir_checks_json": str(out_dir / "casimir_checks_4d.json"),
        "block_profile_csv": str(profile_csv),
        "crossing_derivatives_csv": str(vectors_csv),
    }
    with (out_dir / "summary_4d.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"4D demo complete: {out_dir}")
    return 0


def load_options(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("sdpb-options must be a JSON object")
    return data


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    """Temporarily change cwd and avoid masking original exceptions on restore."""
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        try:
            os.chdir(old_cwd)
        except Exception as exc:
            if sys.exc_info()[0] is None:
                raise
            print(f"Warning: failed to restore cwd to {old_cwd}: {exc}", file=sys.stderr)


def resolve_default_pycftboot_dir(script_dir: Path) -> Path:
    """Pick a sensible default vendor location across repo layouts."""
    candidates = [
        script_dir.parent / "Bootstrap" / "ising3d" / "vendor" / "pycftboot",
        script_dir.parent / "bootstrap" / "vendor" / "pycftboot",
        script_dir.parent / "test" / "bootstrap" / "vendor" / "pycftboot",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def import_pycftboot() -> Any:
    script_dir = Path(__file__).resolve().parent
    default_dir = resolve_default_pycftboot_dir(script_dir)
    pycftboot_dir = Path(os.environ.get("PYCFTBOOT_DIR", str(default_dir))).resolve()
    if not pycftboot_dir.is_dir():
        raise RuntimeError(f"PYCFTBOOT_DIR not found: {pycftboot_dir}")
    required_files = ("bootstrap.py", "common.py")
    missing = [name for name in required_files if not (pycftboot_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"PYCFTBOOT_DIR is missing required files ({', '.join(missing)}): {pycftboot_dir}"
        )

    old_sys_path = list(sys.path)
    try:
        with pushd(pycftboot_dir):
            sys.path.insert(0, str(pycftboot_dir))
            import bootstrap  # type: ignore
    finally:
        sys.path = old_sys_path
    return bootstrap


def sanitize_sdpb_options(bootstrap: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Filter user JSON options to those accepted by this PyCFTBoot+SDPB version."""
    valid = set(getattr(bootstrap, "sdpb_options", []))
    if not valid:
        return options

    filtered: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in options.items():
        if key in valid:
            filtered[key] = value
        else:
            ignored.append(key)
    if ignored:
        print("Ignoring unsupported SDPB options:", ", ".join(sorted(ignored)))
    return filtered


# ===================== 3D section =====================
# In 3D, conformal blocks have no closed-form _2F_1 expression.  PyCFTBoot
# computes them via Zamolodchikov-like radial recursion (Kos, Poland,
# Simmons-Duffin, 1406.4858) and then sets up the SDP for SDPB to solve.
# The workflow is: block table -> convolution -> SDP -> bisection.


def run_3d(args: argparse.Namespace) -> int:
    """Run one 3D SDPB bisection bound and write a JSON record.

    The bound is on the leading Z2-even scalar dimension Delta_epsilon
    at a fixed external dimension Delta_sigma.  The result converges
    toward Delta_epsilon ~ 1.4126 for the 3D Ising CFT as truncation
    is increased.
    """
    if args.dim <= 0:
        raise ValueError("--dim must be > 0")
    if args.k_max < 0 or args.l_max < 0 or args.m_max < 0 or args.n_max < 0:
        raise ValueError("k/l/m/n truncation parameters must be >= 0")
    if not (0 < args.cutoff < 1):
        raise ValueError("--cutoff must be in (0, 1)")
    if args.lower >= args.upper:
        raise ValueError("--lower must be < --upper")
    if args.tol <= 0:
        raise ValueError("--tol must be > 0")
    if args.cutoff < LOCAL_CUTOFF_STABLE_MIN or args.cutoff > LOCAL_CUTOFF_STABLE_MAX:
        print(
            "Warning: cutoff="
            f"{args.cutoff} is outside the locally validated stable window "
            f"[{LOCAL_CUTOFF_STABLE_MIN}, {LOCAL_CUTOFF_STABLE_MAX}]. "
            "At higher truncation this may produce NaN coefficients in "
            "PyCFTBoot XML generation.",
            file=sys.stderr,
        )

    try:
        bootstrap = import_pycftboot()
    except Exception as exc:
        raise RuntimeError(
            "PyCFTBoot module 'bootstrap' is required for mode=3d. "
            "Check PYCFTBOOT_DIR and dependencies."
        ) from exc

    if args.sdpb_path:
        bootstrap.sdpb_path = args.sdpb_path
    if args.mpirun_path:
        bootstrap.mpirun_path = args.mpirun_path
    if args.mpirun_np:
        bootstrap.mpirun_np = int(args.mpirun_np)

    bootstrap.cutoff = args.cutoff

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build conformal block table via Zamolodchikov-like recursion.
    table = bootstrap.ConformalBlockTable(args.dim, args.k_max, args.l_max, args.m_max, args.n_max)
    # Step 2: Convolve blocks into crossing-derivative vectors.
    conv = bootstrap.ConvolvedBlockTable(table)
    # Step 3: Build SDP at fixed Delta_sigma.
    sdp = bootstrap.SDP(args.delta_sigma, conv)

    for key, value in sanitize_sdpb_options(bootstrap, load_options(args.sdpb_options)).items():
        sdp.set_option(key, value)

    # Step 4: Bisect over the Z2-even scalar gap (channel 0).
    # SDPB checks feasibility at each trial gap; the bound is the largest
    # gap that cannot be excluded by crossing + unitarity.
    bound = float(sdp.bisect(args.lower, args.upper, args.tol, 0, name=args.name))

    out_json = out_dir / "bound_3d_single_point.json"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "delta_sigma": args.delta_sigma,
                "bound_delta_even": bound,
                "dim": args.dim,
                "k_max": args.k_max,
                "l_max": args.l_max,
                "m_max": args.m_max,
                "n_max": args.n_max,
                "cutoff": args.cutoff,
                "lower": args.lower,
                "upper": args.upper,
                "tol": args.tol,
                "name": args.name,
            },
            handle,
            indent=2,
        )

    print(f"3D demo complete: {out_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI with notebook-aligned 2D/4D/3D subcommands."""
    p = argparse.ArgumentParser(description="Conformal blocks and bootstrap demo (production-ready streamlined)")
    sub = p.add_subparsers(dest="mode", required=True)

    p2 = sub.add_parser("2d", help="Run the 2D global block section")
    p2.add_argument("--delta-phi", type=float, default=0.125)
    p2.add_argument("--precision", type=int, default=70)
    p2.add_argument("--x-min", type=float, default=0.05)
    p2.add_argument("--x-max", type=float, default=0.45)
    p2.add_argument("--x-step", type=float, default=0.01)
    p2.add_argument("--weight-min", type=float, default=0.0)
    p2.add_argument("--weight-max", type=float, default=6.0)
    p2.add_argument("--weight-step", type=float, default=1.0)
    p2.add_argument("--gap-scan-weight-max", type=float, default=8.0)
    p2.add_argument("--gap-scan-step", type=float, default=0.5)
    p2.add_argument("--out-dir", type=Path, default=Path("streamlined_output_main/2d"))

    p4 = sub.add_parser("4d", help="Run the 4D block/Casimir section")
    p4.add_argument("--delta-phi", type=float, default=1.0)
    p4.add_argument("--precision", type=int, default=90)
    p4.add_argument("--x-min", type=float, default=0.05)
    p4.add_argument("--x-max", type=float, default=0.95)
    p4.add_argument("--x-step", type=float, default=0.01)
    p4.add_argument("--diag-eps", type=float, default=1e-3)
    p4.add_argument("--derivative-degree", type=int, default=3)
    p4.add_argument("--out-dir", type=Path, default=Path("streamlined_output_main/4d"))

    p3 = sub.add_parser("3d", help="Run the 3D SDPB single-point section")
    p3.add_argument("--delta-sigma", type=float, default=0.518)
    p3.add_argument("--dim", type=int, default=3)
    p3.add_argument("--k-max", type=int, default=25)
    p3.add_argument("--l-max", type=int, default=20)
    p3.add_argument("--m-max", type=int, default=1)
    p3.add_argument("--n-max", type=int, default=4)
    p3.add_argument("--cutoff", type=float, default=0.20)
    p3.add_argument("--lower", type=float, default=1.0)
    p3.add_argument("--upper", type=float, default=2.0)
    p3.add_argument("--tol", type=float, default=1e-4)
    p3.add_argument("--name", type=str, default="ising3d_main_demo")
    p3.add_argument("--sdpb-options", type=Path, default=None)
    p3.add_argument("--sdpb-path", type=str, default=None)
    p3.add_argument("--mpirun-path", type=str, default=None)
    p3.add_argument("--mpirun-np", type=int, default=None)
    p3.add_argument("--out-dir", type=Path, default=Path("streamlined_output_main/3d"))

    return p


def main() -> int:
    args = build_parser().parse_args()
    handlers = {"2d": run_2d, "4d": run_4d, "3d": run_3d}
    return handlers[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
