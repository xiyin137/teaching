#!/usr/bin/env python3
"""Streamlined 3D conformal bootstrap demo using PyCFTBoot + SDPB.

Background
----------
The conformal bootstrap derives rigorous bounds on CFT data (operator
dimensions and OPE coefficients) from crossing symmetry and unitarity,
without assuming a Lagrangian.  The key steps are:

1. **Block tables** -- Conformal blocks in 3D have no closed-form expression.
   PyCFTBoot computes them numerically via the Zamolodchikov-like radial
   recursion of Kos, Poland, Simmons-Duffin (1406.4858).
   ``ConformalBlockTable(dim, k_max, l_max, m_max, n_max)`` builds a table
   of blocks truncated at radial order k_max, max spin l_max, and derivative
   orders (m_max, n_max).

2. **Convolution** -- ``ConvolvedBlockTable`` repackages the blocks into the
   form needed for the crossing equation (essentially computing the
   crossing-derivative vectors F_{m,n}).

3. **SDP setup** -- ``SDP(delta_sigma, convolved)`` builds the semidefinite
   program: find a linear functional alpha such that alpha(F_0) > 0 and
   alpha(F_{Delta,ell}) >= 0 for all allowed (Delta, ell).  If such alpha
   exists, the assumed gap is *excluded*.

4. **Bisection** -- ``sdp.bisect(lower, upper, tol, channel)`` binary-searches
   over the gap in the Z2-even scalar channel (channel=0) to find the largest
   gap that is *not* excluded.  The result is an upper bound:
   Delta_epsilon <= bound.

For the 3D Ising CFT at Delta_sigma ~ 0.518, this bound converges toward
Delta_epsilon ~ 1.4126 as truncation is increased.

Usage
-----
  python3 3D_conformal_bootstrap_demo_streamlined.py
  python3 3D_conformal_bootstrap_demo_streamlined.py --delta-sigma 0.518 --k-max 25 --l-max 20

Parameter glossary
------------------
  delta-sigma    external scalar dimension (sigma field in the Ising model)
  dim            spacetime dimension (3)
  k-max          radial expansion truncation order
  l-max          maximum spin included in the block table
  m-max, n-max   number of z, zbar derivatives at the crossing-symmetric point
  cutoff         PyCFTBoot pole-selection threshold (stable in [0.15, 0.20])
  lower, upper   bisection search range for Delta_epsilon
  tol            bisection tolerance

References
----------
  - Kos, Poland, Simmons-Duffin, 1406.4858 (PyCFTBoot algorithm)
  - Simmons-Duffin, 1502.02033 (SDPB solver)
  - El-Showk et al., 1203.6064 (3D Ising bootstrap bounds)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOCAL_CUTOFF_STABLE_MIN = 0.15
LOCAL_CUTOFF_STABLE_MAX = 0.20


def parse_args() -> argparse.Namespace:
    """CLI parameters for truncation, bisection, and executable paths."""
    parser = argparse.ArgumentParser(description="3D Ising single-point bootstrap demo")
    parser.add_argument("--delta-sigma", type=float, default=0.518)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--l-max", type=int, default=20)
    parser.add_argument("--m-max", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=0.20)

    parser.add_argument("--lower", type=float, default=1.0)
    parser.add_argument("--upper", type=float, default=2.0)
    parser.add_argument("--tol", type=float, default=1e-4)

    parser.add_argument("--name", type=str, default="ising3d_single_demo")
    parser.add_argument("--sdpb-options", type=Path, default=None)
    parser.add_argument("--sdpb-path", type=str, default=None)
    parser.add_argument("--mpirun-path", type=str, default=None)
    parser.add_argument("--mpirun-np", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("streamlined_output_3d/demo_result.json"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric ranges early for clearer CLI errors."""
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


def warn_cutoff_stability(cutoff: float) -> None:
    """Warn when cutoff is outside the locally validated stable window."""
    if cutoff < LOCAL_CUTOFF_STABLE_MIN or cutoff > LOCAL_CUTOFF_STABLE_MAX:
        print(
            "Warning: cutoff="
            f"{cutoff} is outside the locally validated stable window "
            f"[{LOCAL_CUTOFF_STABLE_MIN}, {LOCAL_CUTOFF_STABLE_MAX}]. "
            "At higher truncation this may produce NaN coefficients in "
            "PyCFTBoot XML generation.",
            file=sys.stderr,
        )


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


def load_options(path: Path | None) -> dict[str, Any]:
    """Load optional SDPB command-line options from JSON."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("sdpb-options must be a JSON object")
    return data


def resolve_default_pycftboot_dir(script_dir: Path) -> Path:
    """Pick a sensible default vendor location across repo layouts."""
    candidates = [
        script_dir / "vendor" / "pycftboot",
        script_dir.parent / "Bootstrap" / "ising3d" / "vendor" / "pycftboot",
        script_dir.parent / "bootstrap" / "vendor" / "pycftboot",
        script_dir.parent / "test" / "bootstrap" / "vendor" / "pycftboot",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def import_pycftboot() -> Any:
    """Import vendored PyCFTBoot even when invoked from another directory."""
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


def main() -> int:
    """Run one 3D Ising bootstrap bound computation and write JSON output."""
    args = parse_args()
    validate_args(args)
    warn_cutoff_stability(args.cutoff)

    try:
        bootstrap = import_pycftboot()
    except Exception as exc:
        raise RuntimeError(
            "Could not import vendored PyCFTBoot ('bootstrap'). "
            "Check PYCFTBOOT_DIR and dependencies."
        ) from exc

    if args.sdpb_path:
        bootstrap.sdpb_path = args.sdpb_path
    if args.mpirun_path:
        bootstrap.mpirun_path = args.mpirun_path
    if args.mpirun_np:
        bootstrap.mpirun_np = int(args.mpirun_np)

    bootstrap.cutoff = args.cutoff

    # Step 1: Build the conformal block table via Zamolodchikov-like recursion.
    # This is the most compute-intensive step; runtime grows with k_max and l_max.
    print("Building conformal blocks...")
    table = bootstrap.ConformalBlockTable(args.dim, args.k_max, args.l_max, args.m_max, args.n_max)

    # Step 2: Convolve blocks into crossing-derivative vectors F_{m,n}(Delta, ell).
    convolved = bootstrap.ConvolvedBlockTable(table)

    # Step 3: Build the SDP for a single external dimension Delta_sigma.
    # This encodes: "assuming a gap Delta_epsilon in the Z2-even scalar channel,
    # is crossing symmetry + unitarity consistent?"
    print("Setting up SDP...")
    sdp = bootstrap.SDP(args.delta_sigma, convolved)
    for key, value in sanitize_sdpb_options(bootstrap, load_options(args.sdpb_options)).items():
        sdp.set_option(key, value)

    # Step 4: Bisect over the gap in channel 0 (Z2-even scalars).
    # At each trial gap, SDPB checks feasibility of the dual SDP.
    # The result is the largest gap that cannot be excluded:
    #   Delta_epsilon <= bound.
    print("Running bisection bound...")
    bound = float(sdp.bisect(args.lower, args.upper, args.tol, 0, name=args.name))

    result = {
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
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Bound for Delta_sigma={args.delta_sigma:.6f}: Delta_even <= {bound:.6f}")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
