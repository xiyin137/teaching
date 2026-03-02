#!/usr/bin/env python3
"""3D Ising mixed-correlator island scan with PyCFTBoot + SDPB.

Background
----------
The single-correlator bootstrap (see Bootstrap/ising3d/) bounds operator
dimensions by studying the four-point function <sigma sigma sigma sigma>.
The mixed-correlator bootstrap goes further: it simultaneously imposes
crossing symmetry on the system of four-point functions

    <sigma sigma sigma sigma>,   <sigma sigma epsilon epsilon>,
    <epsilon epsilon epsilon epsilon>,

where sigma (Z2-odd, dimension Delta_sigma) and epsilon (Z2-even, dimension
Delta_epsilon) are the two relevant scalars of the 3D Ising CFT.

Crossing + unitarity for this system yields a set of sum rules with
7 independent channels:
  - 5 matrix channels  (even spin, Z2-even exchange):
      mat1..mat5 encode 2x2 matrix positivity conditions mixing the
      <ss|ss>, <ss|ee>, <ee|ee> OPE data.
  - 2 vector channels  (Z2-odd exchange):
      vec_odd_even (even spin) and vec_odd_odd (odd spin) encode the
      <se|se> crossing constraints.

For a given (Delta_sigma, Delta_epsilon), the code checks whether the full
system of sum rules is feasible (i.e., admits a solution consistent with
unitarity). Points where it is feasible are "allowed"; points where no
solution exists are "excluded". Scanning a grid of (Delta_sigma, Delta_epsilon)
values maps out the famous "Ising island" -- a tiny allowed region that
pins down the 3D Ising critical exponents with high precision.

This driver
-----------
For each grid point (Delta_sigma, Delta_epsilon), the pipeline is:

1. Build three ConformalBlockTable objects:
   - base table (delta12=delta34=0): blocks for <OO|OO> with identical externals.
   - mixed_table_a (delta12 = Delta_e - Delta_s, delta34 = Delta_s - Delta_e):
     blocks for the <sigma epsilon|...> channel.
   - mixed_table_b (delta12 = Delta_s - Delta_e, delta34 = Delta_s - Delta_e):
     blocks for the <epsilon sigma|...> channel.

2. Build 5 ConvolvedBlockTable objects from those block tables (the
   crossing-derivative vectors F_{m,n} needed by the SDP).

3. Construct the SDP with the 7-channel mixed vector_types and set:
   - a Z2-even scalar assumption (tutorial or isolated-epsilon mode),
   - unitarity bound on Z2-odd scalars,
   - a single relevant Z2-odd scalar pinned at Delta_sigma.

4. Call iterate() to check feasibility via SDPB.

Output classifies each point as:
  - allowed:  feasibility found,
  - excluded: infeasible under the assumptions,
  - error:    execution failed.

Usage
-----
  python3 ising3d_mixed_island_scan.py \\
      --sigma-start 0.517 --sigma-end 0.521 --sigma-step 0.001 \\
      --epsilon-start 1.410 --epsilon-end 1.435 --epsilon-step 0.005 \\
      --out-dir results/test

Parameter glossary
------------------
  sigma-start/end/step   scan range for Delta_sigma
  epsilon-start/end/step scan range for Delta_epsilon
  dim                    spacetime dimension (3)
  k-max                  radial expansion truncation order
  l-max                  maximum spin in block tables
  m-max, n-max           number of z, zbar derivatives at the crossing point
  cutoff                 PyCFTBoot pole-selection threshold (stable in [0.15, 0.20])
  dual-error-threshold   SDPB convergence criterion for feasibility checks
  even-scalar-assumption
                         tutorial: gap starts at Delta_epsilon
                         isolated_epsilon: pin epsilon at Delta_epsilon and
                         set remaining Z2-even scalar gap at dim

References
----------
  - Kos, Poland, Simmons-Duffin, 1406.4858 (PyCFTBoot algorithm)
  - Kos, Poland, Simmons-Duffin, Vichi, 1603.04436 (mixed-correlator island)
  - Simmons-Duffin, 1502.02033 (SDPB solver)
"""

from __future__ import annotations

import argparse
import copy
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


@dataclass
class IslandPointResult:
    """One grid-point feasibility result."""

    delta_sigma: float
    delta_epsilon: float
    allowed: bool | None
    elapsed_sec: float
    status: str
    error: str
    point_dir: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for mixed-correlator island scanning."""
    parser = argparse.ArgumentParser(description="3D Ising mixed-correlator island scan")

    parser.add_argument("--sigma-start", type=float, required=True, help="scan start for Delta_sigma")
    parser.add_argument("--sigma-end", type=float, required=True, help="scan end for Delta_sigma")
    parser.add_argument("--sigma-step", type=float, required=True, help="scan step for Delta_sigma")

    parser.add_argument("--epsilon-start", type=float, required=True, help="scan start for Delta_epsilon")
    parser.add_argument("--epsilon-end", type=float, required=True, help="scan end for Delta_epsilon")
    parser.add_argument("--epsilon-step", type=float, required=True, help="scan step for Delta_epsilon")

    parser.add_argument("--dim", type=int, default=3, help="spacetime dimension")
    parser.add_argument("--k-max", type=int, default=25, help="radial truncation parameter k_max")
    parser.add_argument("--l-max", type=int, default=20, help="maximum spin in block tables")
    parser.add_argument("--m-max", type=int, default=2, help="m_max in block tables")
    parser.add_argument("--n-max", type=int, default=6, help="n_max in block tables")
    parser.add_argument("--cutoff", type=float, default=0.15, help="PyCFTBoot cutoff parameter")
    parser.add_argument(
        "--even-scalar-assumption",
        type=str,
        choices=("tutorial", "isolated_epsilon"),
        default="tutorial",
        help=(
            "Z2-even spin-0 assumption: "
            "'tutorial' sets gap>=Delta_epsilon; "
            "'isolated_epsilon' pins epsilon at Delta_epsilon and sets "
            "remaining Z2-even scalar gap>=dim."
        ),
    )

    parser.add_argument(
        "--dual-error-threshold",
        type=float,
        default=1e-15,
        help="SDPB/PyCFTBoot dualErrorThreshold override for mixed feasibility checks",
    )

    parser.add_argument("--sdpb-options", type=Path, default=None, help="JSON file with SDPB options")
    parser.add_argument("--sdpb-path", type=str, default=None, help="override SDPB executable path")
    parser.add_argument("--mpirun-path", type=str, default=None, help="override mpirun executable path")
    parser.add_argument("--mpirun-np", type=int, default=None, help="override MPI process count")

    parser.add_argument("--name-prefix", type=str, default="ising3d_mixed", help="prefix for generated SDPB names")
    parser.add_argument("--max-points", type=int, default=1200, help="hard limit on total grid points")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory root for this run")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric arguments before expensive computation."""
    if args.sigma_step <= 0 or args.epsilon_step <= 0:
        raise ValueError("sigma/epsilon step sizes must be > 0")
    if args.sigma_start > args.sigma_end:
        raise ValueError("--sigma-start must be <= --sigma-end")
    if args.epsilon_start > args.epsilon_end:
        raise ValueError("--epsilon-start must be <= --epsilon-end")
    if args.dim <= 0:
        raise ValueError("--dim must be > 0")
    if args.k_max < 0 or args.l_max < 0 or args.m_max < 0 or args.n_max < 0:
        raise ValueError("k/l/m/n truncation parameters must be >= 0")
    if not (0 <= args.cutoff < 1):
        raise ValueError("--cutoff must be in [0, 1)")
    if args.max_points <= 0:
        raise ValueError("--max-points must be > 0")


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    """Temporarily switch directories without masking active exceptions."""
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


def warn_cutoff_stability(cutoff: float) -> None:
    """Warn when cutoff is outside locally validated stable values."""
    if cutoff < LOCAL_CUTOFF_STABLE_MIN or cutoff > LOCAL_CUTOFF_STABLE_MAX:
        print(
            "Warning: cutoff="
            f"{cutoff} is outside the locally validated stable window "
            f"[{LOCAL_CUTOFF_STABLE_MIN}, {LOCAL_CUTOFF_STABLE_MAX}]. "
            "At higher truncation this may produce NaN coefficients in XML generation.",
            file=sys.stderr,
        )


def _prepend_executable_parent_to_path(path_value: str | None) -> None:
    """Prepend parent directory of an executable path to PATH if needed."""
    if not path_value:
        return
    parent = str(Path(path_value).expanduser().resolve().parent)
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    if parent not in entries:
        os.environ["PATH"] = f"{parent}{os.pathsep}{current}" if current else parent


def import_pycftboot() -> Any:
    """Import vendored PyCFTBoot module from local repository."""
    script_dir = Path(__file__).resolve().parent
    default_vendor = script_dir.parent / "ising3d" / "vendor" / "pycftboot"
    pycftboot_dir = Path(os.environ.get("PYCFTBOOT_DIR", str(default_vendor))).resolve()

    if not pycftboot_dir.is_dir():
        raise RuntimeError(f"PYCFTBOOT_DIR not found: {pycftboot_dir}")

    required = ("bootstrap.py", "common.py")
    missing = [name for name in required if not (pycftboot_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"PYCFTBOOT_DIR missing required files ({joined}): {pycftboot_dir}")

    old_sys_path = list(sys.path)
    try:
        with pushd(pycftboot_dir):
            sys.path.insert(0, str(pycftboot_dir))
            import bootstrap  # type: ignore
    finally:
        sys.path = old_sys_path
    return bootstrap


def frange(start: float, end: float, step: float) -> list[float]:
    """Inclusive floating-point range with endpoint tolerance."""
    values: list[float] = []
    x = start
    eps = abs(step) * 1e-9
    while x <= end + eps:
        values.append(round(x, 10))
        x += step
    return values


def sigma_tag(value: float) -> str:
    """Convert sigma value to filesystem-safe tag."""
    return f"{value:.5f}".replace(".", "")


def epsilon_tag(value: float) -> str:
    """Convert epsilon value to filesystem-safe tag."""
    return f"{value:.5f}".replace(".", "")


def load_sdpb_options(path: Path | None) -> dict[str, Any]:
    """Load optional SDPB option overrides from JSON."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("--sdpb-options must contain a JSON object")
    return data


def sanitize_sdpb_options(bootstrap: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Filter unsupported SDPB options for current PyCFTBoot/SDPB version."""
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
        print(f"Ignoring unsupported SDPB options: {', '.join(sorted(ignored))}")
    return filtered


def build_mixed_vector_types() -> list[list[Any]]:
    """Construct mixed-correlator vector types for the 3D Ising Z2 system.

    The mixed system <ss|ss>, <ss|ee>, <ee|ee> (s=sigma, e=epsilon) has three
    crossing-symmetry channels, each built from a different set of OPE data:

    1. Z2-even exchange, even spin (matrix channel):
       Operators appearing in both sigma x sigma and epsilon x epsilon OPEs.
       The 5 matrices mat1..mat5 encode 2x2 positive-semidefinite constraints
       mixing the ss and ee OPE coefficients. Each matrix entry is a 4-element
       list [coefficient, convolved_table_index, parity_flag, extra_flag]
       specifying which convolved block table contributes.

    2. Z2-odd exchange, even spin (vector channel):
       Operators in the sigma x epsilon OPE with even spin.

    3. Z2-odd exchange, odd spin (vector channel):
       Operators in the sigma x epsilon OPE with odd spin.

    These match exactly the PyCFTBoot tutorial (tutorial.py, lines ~129-180).
    """
    # --- Matrix channels: Z2-even operators with even spin ---
    # Each mat_i is a 2x2 matrix of [coeff, table_idx, parity, type] entries.
    # The rows/columns index (sigma, epsilon) external pairs.
    mat1 = [[[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    mat2 = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [1, 0, 1, 1]]]
    mat3 = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    mat4 = [[[0, 0, 0, 0], [0.5, 0, 0, 1]], [[0.5, 0, 0, 1], [0, 0, 0, 0]]]
    mat5 = [[[0, 1, 0, 0], [0.5, 1, 0, 1]], [[0.5, 1, 0, 1], [0, 1, 0, 0]]]

    vec_even_even = [mat1, mat2, mat3, mat4, mat5]

    # --- Vector channels: Z2-odd operators ---
    # Each entry is [coeff, table_idx, parity, type] for a single constraint row.
    # vec_odd_even: even-spin Z2-odd (from sigma x epsilon OPE).
    vec_odd_even = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 0], [1, 2, 0, 0], [-1, 3, 0, 0]]
    # vec_odd_odd: odd-spin Z2-odd (from sigma x epsilon OPE).
    vec_odd_odd = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 0], [-1, 2, 0, 0], [1, 3, 0, 0]]

    return [
        [vec_even_even, 0, "z2-even-l-even"],   # matrix channel, even spin
        [vec_odd_even, 0, "z2-odd-l-even"],      # vector channel, even spin
        [vec_odd_odd, 1, "z2-odd-l-odd"],         # vector channel, odd spin
    ]


def build_convolved_tables(bootstrap: Any, tab_ref: Any, tab_mix_a: Any, tab_mix_b: Any) -> list[Any]:
    """Build the 5 convolved block tables used by the mixed vector system.

    The convolution repackages raw conformal blocks into crossing-derivative
    vectors F_{m,n}(Delta, ell). The 5 tables correspond to:
      [0] ConvolvedBlockTable(base)              -- F- from <ss|ss> / <ee|ee>
      [1] ConvolvedBlockTable(base, sym=True)    -- F+ from <ss|ss> / <ee|ee>
      [2] ConvolvedBlockTable(mixed_a)           -- F- from <se|...>
      [3] ConvolvedBlockTable(mixed_a, sym=True) -- F+ from <se|...>
      [4] ConvolvedBlockTable(mixed_b)           -- from <es|...>

    The vector_types matrices index into this list via their table_idx field.
    """
    return [
        bootstrap.ConvolvedBlockTable(tab_ref),                   # index 0
        bootstrap.ConvolvedBlockTable(tab_ref, symmetric=True),   # index 1
        bootstrap.ConvolvedBlockTable(tab_mix_a),                 # index 2
        bootstrap.ConvolvedBlockTable(tab_mix_a, symmetric=True), # index 3
        bootstrap.ConvolvedBlockTable(tab_mix_b),                 # index 4
    ]


def print_execution_context(args: argparse.Namespace, bootstrap: Any, sdpb_options: dict[str, Any]) -> None:
    """Print reproducibility-critical execution context."""
    print("=== Mixed Ising Island Context ===")
    print("Conformal blocks per grid point:")
    print(
        f"  Base table: ConformalBlockTable(dim={args.dim}, k={args.k_max}, l={args.l_max}, m={args.m_max}, n={args.n_max})"
    )
    print("  Mixed tables: two additional ConformalBlockTable instances with nonzero delta12/delta34")
    print("Convolution and SDP:")
    print("  5 convolved tables -> SDP([Delta_sigma, Delta_epsilon], table_list, vector_types=...) ")
    if args.even_scalar_assumption == "tutorial":
        print(
            "  iterate() assumptions: z2-even scalar gap >= Delta_epsilon "
            "(tutorial mode), single relevant z2-odd scalar at Delta_sigma"
        )
    else:
        print(
            "  iterate() assumptions: pin epsilon at Delta_epsilon and set "
            "remaining z2-even scalar gap >= dim (isolated_epsilon mode), "
            "single relevant z2-odd scalar at Delta_sigma"
        )
    print("Executable chain:")
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


def run_point(
    args: argparse.Namespace,
    bootstrap: Any,
    sdpb_options: dict[str, Any],
    vector_types: list[list[Any]],
    delta_sigma: float,
    delta_epsilon: float,
) -> IslandPointResult:
    """Evaluate one (Delta_sigma, Delta_epsilon) point and return feasibility."""
    s_tag = sigma_tag(delta_sigma)
    e_tag = epsilon_tag(delta_epsilon)
    point_name = f"{args.name_prefix}_s{s_tag}_e{e_tag}"
    point_dir = args.out_dir / f"sigma_{s_tag}_epsilon_{e_tag}"
    point_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    allowed_value: bool | None = None
    status = "ok"
    error = ""

    try:
        with pushd(point_dir):
            # Step 1: Compute the external dimension differences needed for
            # mixed block tables. In the mixed system the external operators
            # have unequal dimensions, so delta12 = Delta_1 - Delta_2 is nonzero.
            delta_es = delta_epsilon - delta_sigma
            delta_se = delta_sigma - delta_epsilon

            # Step 2: Build three conformal block tables via Zamolodchikov-like
            # radial recursion. This is the most compute-intensive step.
            #
            # base_table: identical externals (delta12=delta34=0), used for
            #   <sigma sigma|O|sigma sigma> and <epsilon epsilon|O|epsilon epsilon>.
            base_table = bootstrap.ConformalBlockTable(
                args.dim, args.k_max, args.l_max, args.m_max, args.n_max
            )
            # mixed_table_a: blocks for <sigma epsilon|O|epsilon sigma> channel
            #   with delta12 = Delta_e - Delta_s, delta34 = Delta_s - Delta_e.
            mixed_table_a = bootstrap.ConformalBlockTable(
                args.dim,
                args.k_max,
                args.l_max,
                args.m_max,
                args.n_max,
                delta_es,
                delta_se,
                odd_spins=True,
            )
            # mixed_table_b: blocks for <epsilon sigma|O|epsilon sigma> channel
            #   with delta12 = delta34 = Delta_s - Delta_e.
            mixed_table_b = bootstrap.ConformalBlockTable(
                args.dim,
                args.k_max,
                args.l_max,
                args.m_max,
                args.n_max,
                delta_se,
                delta_se,
                odd_spins=True,
            )

            # Step 3: Convolve to get the crossing-derivative vectors F_{m,n}.
            convolved_tables = build_convolved_tables(
                bootstrap, base_table, mixed_table_a, mixed_table_b
            )

            # Step 4: Build the SDP. The mixed system passes a list
            # [Delta_sigma, Delta_epsilon] as external dimensions, along with
            # the 5 convolved tables and the 7-channel vector_types.
            sdp = bootstrap.SDP(
                [delta_sigma, delta_epsilon],
                convolved_tables,
                vector_types=copy.deepcopy(vector_types),
            )

            for key, value in sdpb_options.items():
                sdp.set_option(key, value)
            if args.dual_error_threshold is not None:
                sdp.set_option("dualErrorThreshold", float(args.dual_error_threshold))

            # Step 5: Set physical assumptions for the 3D Ising model.
            #
            # Z2-even sector:
            # - tutorial mode: gap starts at Delta_epsilon (PyCFTBoot tutorial).
            # - isolated_epsilon mode: pin epsilon at Delta_epsilon, then set
            #   gap for remaining Z2-even scalars at dim.
            if args.even_scalar_assumption == "tutorial":
                sdp.set_bound([0, "z2-even-l-even"], delta_epsilon)
            elif args.even_scalar_assumption == "isolated_epsilon":
                sdp.set_bound([0, "z2-even-l-even"], float(args.dim))
                sdp.add_point([0, "z2-even-l-even"], delta_epsilon)
            else:
                raise ValueError(
                    "Unknown --even-scalar-assumption: "
                    f"{args.even_scalar_assumption}"
                )

            # Z2-odd sector: unitarity gap + one pinned relevant scalar.
            sdp.set_bound([0, "z2-odd-l-even"], float(args.dim))
            sdp.add_point([0, "z2-odd-l-even"], delta_sigma)

            # Step 6: Check feasibility. iterate() writes PMP XML, calls
            # pmp2sdp, then sdpb. Returns True if the point is allowed
            # (primal feasible) or False if excluded (dual feasible).
            allowed_value = bool(sdp.iterate(name=point_name))
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - t0
    return IslandPointResult(
        delta_sigma=delta_sigma,
        delta_epsilon=delta_epsilon,
        allowed=allowed_value,
        elapsed_sec=elapsed,
        status=status,
        error=error,
        point_dir=str(point_dir),
    )


def main() -> int:
    """Run mixed-correlator island scan and write aggregate outputs.

    The main loop iterates over a rectangular grid in (Delta_sigma, Delta_epsilon)
    space. At each point, it builds conformal block tables, constructs the mixed
    SDP, and calls SDPB to check feasibility. Results are collected into CSV and
    JSON files for downstream plotting with plot_ising3d_mixed_island.py.
    """
    args = parse_args()
    validate_args(args)
    warn_cutoff_stability(args.cutoff)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # PyCFTBoot resolves sdpb/mpirun during module import via PATH lookup.
    # Ensure user-provided overrides are visible before importing bootstrap.py.
    _prepend_executable_parent_to_path(args.sdpb_path or os.environ.get("SDPB_PATH"))
    _prepend_executable_parent_to_path(args.mpirun_path or os.environ.get("MPIRUN_PATH"))

    # Import the vendored PyCFTBoot library and configure executable paths.
    try:
        bootstrap = import_pycftboot()
    except Exception as exc:
        raise RuntimeError(
            "Failed to import vendored PyCFTBoot module 'bootstrap'. "
            "Check PYCFTBOOT_DIR and dependencies."
        ) from exc

    if args.sdpb_path:
        bootstrap.sdpb_path = args.sdpb_path
    if args.mpirun_path:
        bootstrap.mpirun_path = args.mpirun_path
    if args.mpirun_np:
        bootstrap.mpirun_np = int(args.mpirun_np)

    # Set the PyCFTBoot cutoff parameter (pole-selection threshold for the
    # Zamolodchikov recursion). Values outside [0.15, 0.20] may produce NaN.
    bootstrap.cutoff = args.cutoff

    sdpb_options = sanitize_sdpb_options(bootstrap, load_sdpb_options(args.sdpb_options))
    print_execution_context(args, bootstrap, sdpb_options)

    # Build the scan grid.
    sigma_values = frange(args.sigma_start, args.sigma_end, args.sigma_step)
    epsilon_values = frange(args.epsilon_start, args.epsilon_end, args.epsilon_step)

    total_points = len(sigma_values) * len(epsilon_values)
    if total_points > args.max_points:
        raise ValueError(
            f"Requested {total_points} points, above --max-points={args.max_points}. "
            "Use coarser steps or raise --max-points explicitly."
        )

    print(
        f"Scanning {len(sigma_values)} sigma values x {len(epsilon_values)} epsilon values "
        f"= {total_points} points"
    )

    # Construct the 7-channel vector types once; deep-copied per SDP call.
    vector_types = build_mixed_vector_types()
    results: list[IslandPointResult] = []

    # Main scan loop: iterate over all (Delta_sigma, Delta_epsilon) grid points.
    idx = 0
    for delta_sigma in sigma_values:
        for delta_epsilon in epsilon_values:
            idx += 1
            result = run_point(
                args,
                bootstrap,
                sdpb_options,
                vector_types,
                delta_sigma,
                delta_epsilon,
            )
            results.append(result)

            if result.status == "ok":
                if result.allowed is True:
                    decision = "allowed"
                elif result.allowed is False:
                    decision = "excluded"
                else:
                    decision = "unknown"
                print(
                    f"[{idx}/{total_points}] sigma={delta_sigma:.5f} epsilon={delta_epsilon:.5f} "
                    f"status=ok decision={decision} time={result.elapsed_sec:.1f}s"
                )
            else:
                print(
                    f"[{idx}/{total_points}] sigma={delta_sigma:.5f} epsilon={delta_epsilon:.5f} "
                    f"status=error error={result.error}"
                )

    # Write results to CSV (one row per grid point) and JSON (with full config).
    csv_path = args.out_dir / "scan_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "delta_sigma",
                "delta_epsilon",
                "allowed",
                "elapsed_sec",
                "status",
                "error",
                "point_dir",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    allowed_count = sum(1 for r in results if r.status == "ok" and r.allowed is True)
    excluded_count = sum(1 for r in results if r.status == "ok" and r.allowed is False)
    error_count = sum(1 for r in results if r.status != "ok")

    json_path = args.out_dir / "scan_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "sigma_start": args.sigma_start,
                    "sigma_end": args.sigma_end,
                    "sigma_step": args.sigma_step,
                    "epsilon_start": args.epsilon_start,
                    "epsilon_end": args.epsilon_end,
                    "epsilon_step": args.epsilon_step,
                    "dim": args.dim,
                    "k_max": args.k_max,
                    "l_max": args.l_max,
                    "m_max": args.m_max,
                    "n_max": args.n_max,
                    "cutoff": args.cutoff,
                    "even_scalar_assumption": args.even_scalar_assumption,
                    "dual_error_threshold": args.dual_error_threshold,
                    "name_prefix": args.name_prefix,
                },
                "summary": {
                    "total_points": total_points,
                    "allowed_points": allowed_count,
                    "excluded_points": excluded_count,
                    "error_points": error_count,
                },
                "results": [asdict(r) for r in results],
            },
            handle,
            indent=2,
        )

    print(f"Completed mixed scan. Results: {csv_path}")
    print(
        f"Summary: allowed={allowed_count}, excluded={excluded_count}, errors={error_count}, total={total_points}"
    )
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
