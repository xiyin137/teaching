import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.linalg as la
import os

# =============================================================================
# ANALYSIS OF 3D SU(2) LATTICE YANG-MILLS DATA
# =============================================================================
#
# This script extracts physical observables from the simulation data:
#
# 1. GLUEBALL MASS via GEVP (Generalized Eigenvalue Problem):
#    - Build correlation matrix C_ij(t) = <O_i(0) O_j(t)> for smeared operators
#    - Solve generalized eigenvalue problem: C(t) v = lambda(t) C(t0) v
#    - Eigenvalues decay as lambda_n(t) ~ exp(-m_n * t) for large t
#    - Largest eigenvalue gives ground state (0++ glueball) mass
#
# 2. STRING TENSION from Wilson loops:
#    - Wilson loop W(R,T) ~ exp(-V(R) * T) for large T
#    - Static quark potential V(R) extracted from: V(R) = log(W(R,T)/W(R,T+1))
#    - For confining theory: V(R) = sigma * R + const (linear potential)
#    - sigma is the string tension (energy per unit length of flux tube)
#
# 3. PHYSICAL RATIO m_G / sqrt(sigma):
#    - Dimensionless ratio comparing glueball mass to string tension scale
#    - Should be ~4-5 for 0++ glueball in 3D SU(2) Yang-Mills
#
# =============================================================================

# --- LOAD DATA ---
# Use script's directory to find data file
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'lattice_data_3d.npz')
data = np.load(data_path)
ops_history = data['ops_history']  # Shape: (n_configs, n_operators, L)
wilson_avg = data['wilson_avg']    # Shape: (R_max, T_max)
L = int(data['L'])
beta = float(data['beta'])

print(f"Loaded Data: Beta={beta}, L={L}, Configs={ops_history.shape[0]}")

# =============================================================================
# GEVP ANALYSIS FOR GLUEBALL MASS
# =============================================================================
#
# The GEVP method uses multiple operators (different smearing levels) to
# improve overlap with physical states and reduce excited state contamination.
#
# Steps:
#   1. Subtract vacuum expectation value (VEV) from each operator
#   2. Build correlation matrix: C_ij(t) = <O_i(0) O_j(t)>
#   3. Symmetrize: C_ij = C_ji (should be symmetric by construction)
#   4. Fold using periodicity: C(t) = [C(t) + C(Nt-t)]/2
#   5. Solve GEVP: C(t) v = lambda(t,t0) C(t0) v
#   6. Extract mass from eigenvalue: m = -log(lambda(t)/lambda(t-1))
#
# =============================================================================

n_meas, n_ops, Nt = ops_history.shape
# Note: Nt = L (lattice size) since we use the z-direction as Euclidean "time"

# Step 1: Subtract VEV to get connected correlator
# <O_i O_j> - <O_i><O_j> isolates the physical signal
vevs = np.mean(ops_history, axis=0)      # Average over configs
vev_per_op = np.mean(vevs, axis=1)       # Average over time slices
ops_sub = np.zeros_like(ops_history)
for k in range(n_ops):
    ops_sub[:, k, :] = ops_history[:, k, :] - vev_per_op[k]

# Step 2: Build correlation matrix C_ij(t) = <O_i(0) O_j(t)>
# Using translation invariance: average over all source positions
C_matrix = np.zeros((Nt, n_ops, n_ops))
for t in range(Nt):
    for i_op in range(n_ops):
        for j_op in range(n_ops):
            # Correlate O_i at position z with O_j at position z+t
            prod = ops_sub[:, i_op, :] * np.roll(ops_sub[:, j_op, :], -t, axis=1)
            C_matrix[t, i_op, j_op] = np.mean(prod)

# Step 3: Symmetrize C_ij(t) = [C_ij(t) + C_ji(t)]/2
# Should already be symmetric, but enforce it numerically
C_matrix = 0.5 * (C_matrix + np.transpose(C_matrix, (0, 2, 1)))

# Step 4: Fold correlator using periodic boundary conditions
# C(t) and C(Nt-t) should be equal; averaging reduces noise
for t in range(1, Nt//2 + 1):
    C_matrix[t] = 0.5 * (C_matrix[t] + C_matrix[Nt-t])

# Step 5: Solve GEVP: C(t) v = lambda(t) C(t0) v
# This is a generalized eigenvalue problem that optimally separates states
# t0 is a reference time; t0=0 or t0=1 are common choices
eig_vals = np.zeros((Nt//2, n_ops))
t0_gevp = 0  # Reference time for GEVP
for t in range(Nt//2):
    try:
        # scipy.linalg.eigh solves A v = lambda B v for symmetric A, B
        evals = la.eigh(C_matrix[t], C_matrix[t0_gevp], eigvals_only=True)
        eig_vals[t, :] = np.sort(evals)[::-1]  # Sort descending (largest first)
    except (np.linalg.LinAlgError, ValueError) as e:
        # GEVP can fail if matrices are singular or not positive definite
        eig_vals[t, :] = np.nan

# Ground state eigenvalue (largest eigenvalue = lightest state)
lambda_0 = eig_vals[:, 0]

# =============================================================================
# FITTING THE CORRELATOR TO EXTRACT MASS
# =============================================================================
#
# On a periodic lattice, the correlator has the form:
#   C(t) ~ A * [exp(-m*t) + exp(-m*(Nt-t))] = A * cosh(m*(t - Nt/2)) * const
#
# For noisy data, we add a constant floor:
#   C(t) = A * cosh(m*(t - Nt/2)) + C
#
# The mass m is extracted from the fit.
#
# =============================================================================

def cosh_noise_model(t, A, m, C):
    """Cosh correlator with noise floor for fitting."""
    return A * np.cosh(m * (t - Nt/2.0)) + C

# Fit range: exclude t=0 (contact terms) and large t (noise dominated)
t_start = 1
t_end = 5

t_data = np.arange(Nt//2)
y_data = lambda_0

fit_success = False
popt = None
try:
    # Fit with bounds: A>0, 0<m<5, 0<C<1
    popt, pcov = curve_fit(cosh_noise_model,
                           t_data[t_start:t_end],
                           y_data[t_start:t_end],
                           p0=[y_data[1], 1.0, 0.01],
                           bounds=([0, 0, 0], [np.inf, 5.0, 1.0]))
    mass_est = popt[1]
    mass_err = np.sqrt(pcov[1,1])  # Error from covariance matrix
    fit_success = True
    print(f"\nGLUEBALL MASS (m_G * a): {mass_est:.4f} +/- {mass_err:.4f}  [lattice units]")
except (RuntimeError, ValueError) as e:
    print(f"Fit failed: {e}")
    mass_est = 0.0

# =============================================================================
# STRING TENSION FROM WILSON LOOPS
# =============================================================================
#
# Wilson loop W(R,T) = <Tr[U_path]> around an R x T rectangle.
# For large T, it decays exponentially: W(R,T) ~ exp(-V(R) * T)
#
# The static quark potential V(R) is extracted from the ratio:
#   V(R) = log(W(R,T) / W(R,T+1))
#
# For a confining theory, V(R) grows linearly at large R:
#   V(R) = sigma * R + V0 + O(1/R)
#
# where sigma is the string tension.
#
# =============================================================================

R_max = wilson_avg.shape[0]
V_R = np.zeros(R_max)
V_R[:] = np.nan

# Extract V(R) from Wilson loop ratios at fixed T=3,4
for r in range(R_max):
    if wilson_avg[r, 3] > 0 and wilson_avg[r, 4] > 0:
        # V(R) = -log(W(R,T+1)/W(R,T)) = log(W(R,T)/W(R,T+1))
        V_R[r] = np.log(wilson_avg[r, 3] / wilson_avg[r, 4])

# Fit V(R) = sigma * R + const for R >= 2 (avoid small-R Coulomb effects)
r_vals = np.arange(1, R_max+1)
mask = np.isfinite(V_R) & (r_vals >= 2)
sigma_a2 = np.nan
popt_s = None
if np.sum(mask) >= 2:
    popt_s, _ = curve_fit(lambda r, s, c: s*r + c, r_vals[mask], V_R[mask])
    sigma_a2 = popt_s[0]  # String tension in lattice units (a^2)
    print(f"STRING TENSION (sigma * a^2): {sigma_a2:.4f}  [lattice units]")

# =============================================================================
# PHYSICAL RATIO: m_glueball / sqrt(sigma)
# =============================================================================
#
# This dimensionless ratio is a universal prediction of the theory,
# independent of the lattice spacing. It can be compared to continuum
# extrapolations and other calculations.
#
# For 3D SU(2) Yang-Mills, m_0++ / sqrt(sigma) ~ 4-5.
#
# =============================================================================

if not np.isnan(sigma_a2) and mass_est > 0:
    ratio = mass_est / np.sqrt(sigma_a2)
    print(f"RATIO m_G/sqrt(sigma): {ratio:.4f}  [dimensionless, expected ~4-5 for 0++ glueball]")

# =============================================================================
# PLOTTING
# =============================================================================

plt.figure(figsize=(12, 5))

# Left panel: Correlator and fit
plt.subplot(1, 2, 1)
plt.plot(t_data, y_data, 'bo', label='Data')
if fit_success:
    fit_x = np.linspace(0, Nt//2, 100)
    plt.plot(fit_x, cosh_noise_model(fit_x, *popt), 'r-', label=f'Fit: $m_G a$ = {mass_est:.3f}')
plt.yscale('log')
plt.ylim(1e-3, 2.0)
plt.title(f'GEVP Correlator (Fit Range: t=[{t_start}, {t_end}])')
plt.xlabel('t/a (lattice units)')
plt.ylabel(r'$\lambda_0(t)$')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

# Right panel: Static potential
plt.subplot(1, 2, 2)
plt.plot(r_vals, V_R, 'bo', label='V(R) data')
if not np.isnan(sigma_a2) and popt_s is not None:
    plt.plot(r_vals, r_vals*sigma_a2 + popt_s[1], 'r--', label=rf'$\sigma a^2$ = {sigma_a2:.3f}')
plt.xlabel('R/a (lattice units)')
plt.ylabel('V(R) a')
plt.title('Static Quark Potential')
plt.legend()
plt.grid(True, alpha=0.3)

# Add dimensionless ratio as text annotation
if not np.isnan(sigma_a2) and mass_est > 0:
    ratio = mass_est / np.sqrt(sigma_a2)
    plt.figtext(0.5, 0.02, f'Dimensionless ratio: $m_G / \\sqrt{{\\sigma}}$ = {ratio:.2f}',
                ha='center', fontsize=11, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave room for bottom text

# Save figure
output_path = os.path.join(script_dir, 'latticeYM3d_analysis.png')
plt.savefig(output_path, dpi=150)
print(f"\nFigure saved to: {output_path}")
plt.show()
