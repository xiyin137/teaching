## 2D Ising Model

**Scripts:**
- `ising2d-simulation.py` — Wolff cluster Monte Carlo, measures Tc and observables
- `ising2d-analysis.py` — Extracts critical exponents via finite-size scaling
- `ising2d-diagnostics.py` — Diagnostic plots

**Output:**
- `ising2d_analysis.png` — Analysis results
- `ising2d_diagnostics.png` — Diagnostic plots

## 3D SU(2) Yang-Mills

**Scripts:**
- `latticeYM3d-simulation.py` — Parallel Metropolis Monte Carlo for lattice gauge theory
- `latticeYM3d-analysis.py` — Extracts glueball mass (GEVP) and string tension (Wilson loops)
- `latticeYM3d-errors.py` — Statistical error analysis (autocorrelation, binning, resampling, jackknife)

**Output:**
- `latticeYM3d_analysis.png` — Glueball correlator fit and static quark potential
- `latticeYM3d_errors.png` — Error analysis and autocorrelation diagnostics
