#!/usr/bin/env python3
"""3D Ising even-sector bootstrap scan using PyCFTBoot + SDPB.

Overview
--------
This driver scans a range of external scalar dimensions (Delta_sigma) and,
for each point, computes an upper bound on the first Z2-even scalar dimension
via SDPB bisection.

Design notes
------------
- Conformal block tables are built once per process (shared across sigma points).
- Each sigma point runs in its own output subdirectory `sigma_<tag>`.
- Optional SDPB options are loaded from JSON and filtered against the currently
  available PyCFTBoot/SDPB option list.
- Failures are recorded per point and surfaced in aggregate CSV/JSON outputs.

Execution pipeline (where files are generated)
----------------------------------------------
1) Conformal blocks:
   - `ConformalBlockTable(dim, k_max, l_max, m_max, n_max)` builds truncated
     derivative/radial data for conformal blocks.
   - `ConvolvedBlockTable(table)` converts these into crossing-ready vectors.
2) SDP preparation:
   - `SDP(delta_sigma, convolved)` creates the optimization object.
   - During `bisect(...)`, PyCFTBoot writes polynomial-matrix problem data and
     prepares SDPB input artifacts in the current working directory.
3) SDPB call:
   - PyCFTBoot calls `pmp2sdp` to prepare SDP files (e.g. `mySDP*`) and then
     launches `sdpb` (optionally through `mpirun`), using configured paths.
   - In local mode those executable paths usually point to Docker-backed
     wrapper scripts from `bin/`.

Typical usage
-------------
Single point:
    python ising3d_even_scan.py \
      --sigma-start 0.518 --sigma-end 0.518 --out-dir results/local

Range scan:
    python ising3d_even_scan.py \
      --sigma-start 0.518 --sigma-end 0.521 --sigma-step 0.001 \
      --out-dir results/local
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

LOCAL_CUTOFF_STABLE_MIN = 0.15
LOCAL_CUTOFF_STABLE_MAX = 0.20


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for scan range, truncation, and execution settings."""
    parser = argparse.ArgumentParser(description="3D Ising even-sector scan")

    # Sigma scan window.
    parser.add_argument("--sigma-start", type=float, required=True, help="scan start for Delta_sigma")
    parser.add_argument("--sigma-end", type=float, required=True, help="scan end for Delta_sigma")
    parser.add_argument("--sigma-step", type=float, default=0.001, help="scan step for Delta_sigma")

    # Block-table truncation parameters.
    parser.add_argument("--dim", type=int, default=3, help="spacetime dimension")
    parser.add_argument("--k-max", type=int, default=25, help="radial truncation parameter k_max")
    parser.add_argument("--l-max", type=int, default=20, help="maximum spin in block table")
    parser.add_argument("--m-max", type=int, default=1, help="m_max in block table")
    parser.add_argument("--n-max", type=int, default=4, help="n_max in block table")
    parser.add_argument("--cutoff", type=float, default=0.20, help="PyCFTBoot cutoff parameter")

    # Bisection window for the even-scalar bound.
    parser.add_argument("--lower", type=float, default=1.0, help="lower bisection bracket")
    parser.add_argument("--upper", type=float, default=2.0, help="upper bisection bracket")
    parser.add_argument("--tol", type=float, default=1e-4, help="bisection tolerance")

    # Optional executable overrides and SDPB options.
    parser.add_argument("--sdpb-options", type=Path, default=None, help="JSON file with SDPB options")
    parser.add_argument("--sdpb-path", type=str, default=None, help="override SDPB executable path")
    parser.add_argument("--mpirun-path", type=str, default=None, help="override mpirun executable path")
    parser.add_argument("--mpirun-np", type=int, default=None, help="override MPI process count")

    # Optional extra check and naming.
    parser.add_argument("--single-relevant-check", action="store_true", help="run single-relevant test after bound")
    parser.add_argument("--name-prefix", type=str, default="ising3d_even", help="prefix for generated SDPB problem names")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory root for this run")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric argument ranges before starting expensive work."""
    if args.sigma_step <= 0:
        raise ValueError("--sigma-step must be > 0")
    if args.sigma_start > args.sigma_end:
        raise ValueError("--sigma-start must be <= --sigma-end")
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
            # Preserve the original exception if one is already being raised.
            if sys.exc_info()[0] is None:
                raise
            print(f"Warning: failed to restore cwd to {old_cwd}: {exc}", file=sys.stderr)


def import_pycftboot() -> Any:
    """Import vendored PyCFTBoot `bootstrap` module robustly.

    PyCFTBoot's module layout expects local file execution (`common.py`, etc.).
    To avoid path issues, this function temporarily switches cwd and sys.path.
    """
    script_dir = Path(__file__).resolve().parent
    pycftboot_dir = Path(
        os.environ.get("PYCFTBOOT_DIR", str(script_dir / "vendor" / "pycftboot"))
    ).resolve()

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


def frange(start: float, end: float, step: float) -> list[float]:
    """Inclusive floating range helper with endpoint tolerance."""
    if step <= 0:
        raise ValueError("sigma-step must be > 0")
    values: list[float] = []
    x = start
    while x <= end + (abs(step) * 1e-9):
        values.append(x)
        x += step
    return values


def sigma_tag(value: float) -> str:
    """Create a filesystem-friendly sigma tag (e.g. 0.51800 -> 051800)."""
    return f"{value:.5f}".replace(".", "")


def load_sdpb_options(path: Path | None) -> dict[str, Any]:
    """Load SDPB options JSON (dictionary expected)."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("sdpb-options JSON must be an object")
    return data


def sanitize_sdpb_options(bootstrap: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Keep only options supported by this PyCFTBoot/SDPB combination."""
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
        ignored_str = ", ".join(sorted(ignored))
        print(f"Ignoring unsupported SDPB options: {ignored_str}")

    return filtered


@dataclass
class ScanResult:
    """Single sigma-point result record written to CSV/JSON outputs."""

    delta_sigma: float
    bound_delta_even: float | None
    single_relevant_allowed: bool | None
    single_relevant_status: str
    elapsed_sec: float
    status: str
    error: str


def print_execution_context(
    args: argparse.Namespace, bootstrap: Any, sdpb_options: dict[str, Any]
) -> None:
    """Print a compact, explicit summary of the generation/execution pipeline.

    This is intentionally verbose enough for reproducibility logs: it shows
    where conformal blocks are generated, how SDPB files are prepared, and
    which executables are expected to be used by PyCFTBoot.
    """
    print("=== Bootstrap Pipeline Context ===")
    print("Conformal blocks:")
    print(
        f"  ConformalBlockTable(dim={args.dim}, k_max={args.k_max}, "
        f"l_max={args.l_max}, m_max={args.m_max}, n_max={args.n_max})"
    )
    print("  ConvolvedBlockTable(table)")
    print("SDP preparation per sigma point:")
    print("  SDP(delta_sigma, convolved)")
    print(
        f"  bisect(lower={args.lower}, upper={args.upper}, tol={args.tol}, "
        f"name_prefix={args.name_prefix})"
    )
    print("Expected executable chain inside PyCFTBoot:")
    print(f"  pmp2sdp from PATH -> {shutil.which('pmp2sdp') or 'NOT FOUND'}")
    print(f"  sdpb_path -> {getattr(bootstrap, 'sdpb_path', 'UNKNOWN')}")
    print(f"  mpirun_path -> {getattr(bootstrap, 'mpirun_path', 'UNKNOWN')}")
    print(f"  mpirun_np -> {getattr(bootstrap, 'mpirun_np', 'DEFAULT')}")
    print(f"PYCFTBOOT_DIR -> {os.environ.get('PYCFTBOOT_DIR', 'UNSET')}")
    print(f"Output root -> {args.out_dir}")
    if sdpb_options:
        print(f"SDPB options ({len(sdpb_options)} keys) -> {sorted(sdpb_options)}")
    else:
        print("SDPB options -> none")
    print("==================================")


def main() -> int:
    """Execute scan, write outputs, and return process status code."""
    args = parse_args()
    validate_args(args)
    warn_cutoff_stability(args.cutoff)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Import PyCFTBoot bootstrap module.
    try:
        bootstrap = import_pycftboot()
    except Exception as exc:
        raise RuntimeError(
            "Failed to import vendored PyCFTBoot module 'bootstrap'. "
            "Check PYCFTBOOT_DIR and local dependencies."
        ) from exc

    # Optional path and execution overrides.
    if args.sdpb_path:
        bootstrap.sdpb_path = args.sdpb_path
    if args.mpirun_path:
        bootstrap.mpirun_path = args.mpirun_path
    if args.mpirun_np:
        bootstrap.mpirun_np = int(args.mpirun_np)

    bootstrap.cutoff = args.cutoff

    sdpb_options = sanitize_sdpb_options(
        bootstrap, load_sdpb_options(args.sdpb_options)
    )
    print_execution_context(args, bootstrap, sdpb_options)

    sigma_values = [
        round(v, 10)
        for v in frange(args.sigma_start, args.sigma_end, args.sigma_step)
    ]

    # Step 1: generate conformal blocks once for the chosen truncation.
    print("Preparing conformal block tables...")
    table = bootstrap.ConformalBlockTable(
        args.dim,
        args.k_max,
        args.l_max,
        args.m_max,
        args.n_max,
    )
    convolved = bootstrap.ConvolvedBlockTable(table)

    results: list[ScanResult] = []

    for delta_sigma in sigma_values:
        point_tag = sigma_tag(delta_sigma)
        point_name = f"{args.name_prefix}_{point_tag}"
        point_dir = args.out_dir / f"sigma_{point_tag}"
        point_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        bound_value: float | None = None
        single_relevant_allowed: bool | None = None
        single_relevant_status = "not_requested"
        if args.single_relevant_check:
            single_relevant_status = "not_run"
        status = "ok"
        error = ""

        # Run each sigma point in its own working directory so generated
        # PMP/SDP artifacts (e.g. mySDP, checkpoints, solutions) are isolated.
        try:
            with pushd(point_dir):
                # Step 2: prepare the SDP object at this external dimension.
                sdp = bootstrap.SDP(delta_sigma, convolved)
                for key, value in sdpb_options.items():
                    sdp.set_option(key, value)

                # Step 3: bisection triggers PyCFTBoot's PMP->SDP preparation and
                # launches SDPB through configured executable paths.
                bound_value = float(
                    sdp.bisect(args.lower, args.upper, args.tol, 0, name=point_name)
                )

                # Optional check used in some workflows to test if only one
                # relevant scalar remains after imposing the bound.
                if args.single_relevant_check:
                    sdp.set_bound(0, args.dim)
                    sdp.add_point(0, bound_value)
                    single_relevant_allowed = bool(sdp.iterate(point_name + "_single"))
                    single_relevant_status = (
                        "allowed" if single_relevant_allowed else "excluded"
                    )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            if args.single_relevant_check and single_relevant_status == "not_run":
                single_relevant_status = "point_failed"

        elapsed = time.time() - t0
        result = ScanResult(
            delta_sigma=delta_sigma,
            bound_delta_even=bound_value,
            single_relevant_allowed=single_relevant_allowed,
            single_relevant_status=single_relevant_status,
            elapsed_sec=elapsed,
            status=status,
            error=error,
        )
        results.append(result)

        print(
            f"sigma={delta_sigma:.5f} status={status} "
            f"bound={bound_value if bound_value is not None else 'NA'} "
            f"time={elapsed:.1f}s"
        )

    # Write aggregate CSV.
    csv_path = args.out_dir / "scan_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "delta_sigma",
                "bound_delta_even",
                "single_relevant_allowed",
                "single_relevant_status",
                "elapsed_sec",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    # Write aggregate JSON.
    json_path = args.out_dir / "scan_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(r) for r in results], handle, indent=2)

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print(f"Completed with {len(failed)} failed points. See {json_path}")
        return 1

    print(f"Completed {len(results)} points. Results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
