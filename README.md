# CFT Teaching Materials

Lecture demos and computational exercises for conformal field theory (CFT) and related topics.

## Contents

### [ConformalBlocks/](ConformalBlocks/)
Conformal block computation and crossing symmetry demos in Python and Mathematica.
Covers 2D (holomorphic factorization), 4D (Dolan-Osborn), and 3D (numerical via PyCFTBoot/SDPB).

### [Bootstrap/](Bootstrap/)
Numerical conformal bootstrap workflows:
- **`ising3d/`** -- Single-correlator 3D Ising bootstrap. Builds block tables, solves the crossing SDP with SDPB, and bisects to find the upper bound on the leading Z2-even scalar dimension.
- **`ising3d_mixed/`** -- Mixed-correlator 3D Ising island scan. Scans a (Delta_sigma, Delta_epsilon) grid with the full 5-matrix + 2-vector sum rule to map the allowed island.

### [MonteCarlo/](MonteCarlo/)
Lattice Monte Carlo simulations:
- **2D Ising model** -- Wolff cluster algorithm, finite-size scaling, critical exponent extraction.
- **3D SU(2) Yang-Mills** -- Metropolis Monte Carlo, glueball mass (GEVP), string tension (Wilson loops), error analysis.

## Prerequisites

- **Python 3.9+** with NumPy, SciPy, Matplotlib
- **PyCFTBoot** (vendored in `Bootstrap/ising3d/vendor/pycftboot/`)
- **SDPB** -- for 3D bootstrap. Docker setup scripts included (`Bootstrap/ising3d/setup_sdpb_local.sh`)
- **Mathematica / WolframScript** (optional) -- for `.wl` and `.nb` demos

## Quick Start

```bash
# 2D/4D conformal block demos (no external solver needed)
cd ConformalBlocks
python3 conformal_blocks_and_bootstrap_demo_streamlined.py 2d
python3 conformal_blocks_and_bootstrap_demo_streamlined.py 4d

# 3D Ising single-correlator bootstrap (requires SDPB)
cd Bootstrap/ising3d
bash setup_python_local.sh
source setup_sdpb_local.sh
bash run_point_local.sh 0.518

# Monte Carlo simulations
cd MonteCarlo
python3 ising2d-simulation.py
python3 latticeYM3d-simulation.py
```
