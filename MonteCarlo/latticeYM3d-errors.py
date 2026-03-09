import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.linalg as la
import os

# =============================================================================
# ERROR ANALYSIS FOR 3D SU(2) LATTICE YANG-MILLS
# =============================================================================
#
# This script performs rigorous statistical error analysis:
#   1. Autocorrelation analysis - check if measurements are independent
#   2. Binning analysis - see how errors scale with bin size
#   3. Block bootstrap - non-parametric error estimation for correlated data
#   4. Block jackknife - bias/error estimates for correlated data
#
# WHY ERROR ANALYSIS MATTERS:
# Monte Carlo measurements are correlated - successive configurations differ
# by only one Metropolis sweep, so they share most of their links. The naive
# error formula sigma/sqrt(N) assumes independent samples and will therefore
# UNDERESTIMATE the true statistical uncertainty.
#
# KEY QUANTITIES:
# - Autocorrelation time tau_int: measures how many sweeps between independent
#   samples. Related to how quickly the Markov chain explores configuration space.
# - Effective independent samples: N_eff = N / (2 * tau_int)
# - True error: sigma / sqrt(N_eff) = sigma * sqrt(2*tau_int) / sqrt(N)
#
# RELATIONSHIP TO n_skip:
# In the simulation, we skip n_skip sweeps between measurements. If n_skip ~ tau_int,
# then successive measurements are approximately independent and tau_int (measured
# in units of measurement index) should be ~1. If n_skip << tau_int, measurements
# are still correlated and tau_int (in measurement index) will be > 1.
#
# =============================================================================

# --- LOAD DATA ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'lattice_data_3d.npz')
data = np.load(data_path)
ops_history = data['ops_history']
wilson_history = data['wilson_history'] if 'wilson_history' in data.files else None
wilson_avg = np.mean(wilson_history, axis=0) if wilson_history is not None else data['wilson_avg']
L = int(data['L'])
beta = float(data['beta'])
GEVP_T0 = int(os.environ.get("YM_GEVP_T0", "0"))
if GEVP_T0 < 0:
    raise ValueError("YM_GEVP_T0 must be >= 0")

# Runtime controls for expensive resampling sections.
PLAQ_BOOTSTRAP_SAMPLES = int(os.environ.get("YM_PLAQ_BOOTSTRAP_SAMPLES", "1000"))
PHYSICS_BOOTSTRAP_SAMPLES = int(os.environ.get("YM_PHYSICS_BOOTSTRAP_SAMPLES", "200"))
PHYSICS_BOOTSTRAP_STRIDE = int(os.environ.get("YM_PHYSICS_BOOTSTRAP_STRIDE", "1"))
PHYSICS_BOOTSTRAP_CFG_LIMIT = int(os.environ.get("YM_PHYSICS_BOOTSTRAP_CFG_LIMIT", "0"))
if PLAQ_BOOTSTRAP_SAMPLES < 10:
    raise ValueError("YM_PLAQ_BOOTSTRAP_SAMPLES must be >= 10")
if PHYSICS_BOOTSTRAP_SAMPLES < 10:
    raise ValueError("YM_PHYSICS_BOOTSTRAP_SAMPLES must be >= 10")
if PHYSICS_BOOTSTRAP_STRIDE < 1:
    raise ValueError("YM_PHYSICS_BOOTSTRAP_STRIDE must be >= 1")
if PHYSICS_BOOTSTRAP_CFG_LIMIT < 0:
    raise ValueError("YM_PHYSICS_BOOTSTRAP_CFG_LIMIT must be >= 0")

n_meas, n_ops, Nt = ops_history.shape
print(f"Loaded Data: Beta={beta}, L={L}, Configs={n_meas}")
print(f"GEVP reference time t0 = {GEVP_T0}")
if wilson_history is None:
    print("Note: wilson_history not found; string-tension uncertainty will be approximate.")

rng = np.random.default_rng(12345)

# =============================================================================
# 1. AUTOCORRELATION ANALYSIS
# =============================================================================

def autocorrelation(x, max_lag=None):
    """
    Compute normalized autocorrelation function C(t) = <x(0)x(t)> / <x(0)^2>.

    For independent samples, C(t>0) ~ 0.
    For correlated samples, C(t) decays exponentially with autocorrelation time.
    """
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4

    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)

    acf = np.zeros(max_lag)
    for t in range(max_lag):
        acf[t] = np.mean(x[:n-t] * x[t:]) / var
    return acf

def integrated_autocorr_time(acf, c=5.0):
    """
    Compute integrated autocorrelation time: tau_int = 0.5 + sum_{t=1}^{W} C(t)

    Uses automatic windowing (Madras & Sokal): W is the first t where t >= c * tau_int(t).
    This avoids including noise from large-t autocorrelation estimates.

    The effective number of independent samples is N_eff = N / (2 * tau_int).
    """
    n = len(acf)
    tau_int = 0.5
    for t in range(1, n):
        tau_int += acf[t]
        # Automatic windowing criterion
        if t >= c * tau_int:
            return tau_int, t
    return tau_int, n-1

# Analyze autocorrelation of the plaquette (summed glueball operator)
plaq_timeseries = np.sum(ops_history[:, 0, :], axis=1)  # Sum over z-slices
acf = autocorrelation(plaq_timeseries, max_lag=min(500, n_meas//4))
tau_int, window = integrated_autocorr_time(acf)

print(f"\n--- AUTOCORRELATION ANALYSIS ---")
print(f"Integrated autocorrelation time: tau_int = {tau_int:.2f}")
print(f"Window used: {window}")
print(f"Effective independent samples: N_eff = {n_meas / (2*tau_int):.1f}")
print(f"Naive vs true error ratio: {np.sqrt(2*tau_int):.2f}x")

# =============================================================================
# 2. BINNING ANALYSIS
# =============================================================================

def binning_analysis(x, max_bin_size=None):
    """
    Compute error as function of bin size.

    For independent data, error is constant.
    For correlated data, error grows until bin_size ~ 2*tau_int, then plateaus.
    The plateau value is the true error.
    """
    n = len(x)
    if max_bin_size is None:
        max_bin_size = n // 10

    bin_sizes = []
    errors = []

    for bin_size in range(1, max_bin_size + 1):
        n_bins = n // bin_size
        if n_bins < 2:
            break
        # Bin the data
        binned = np.array([np.mean(x[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
        # Standard error of binned mean
        err = np.std(binned, ddof=1) / np.sqrt(n_bins)
        bin_sizes.append(bin_size)
        errors.append(err)

    return np.array(bin_sizes), np.array(errors)

bin_sizes, bin_errors = binning_analysis(plaq_timeseries)
naive_error = np.std(plaq_timeseries, ddof=1) / np.sqrt(n_meas)

print(f"\n--- BINNING ANALYSIS ---")
print(f"Naive error (bin=1): {naive_error:.6f}")
print(f"Plateau error (bin={bin_sizes[-1]}): {bin_errors[-1]:.6f}")
print(f"Error inflation factor: {bin_errors[-1]/naive_error:.2f}x")

# =============================================================================
# 3. BLOCK RESAMPLING FOR CORRELATED DATA
# =============================================================================
# Standard iid bootstrap/jackknife underestimate uncertainties for Markov chains.
# We use block resampling with block size set by the measured autocorrelation.

def _make_blocks(n, block_size):
    n_blocks = n // block_size
    trimmed = n_blocks * block_size
    if n_blocks < 2:
        return None, 0
    blocks = np.arange(trimmed).reshape(n_blocks, block_size)
    return blocks, trimmed

def block_bootstrap_indices(n, block_size):
    blocks, _ = _make_blocks(n, block_size)
    if blocks is None:
        return np.arange(n)
    n_blocks = blocks.shape[0]
    pick = rng.integers(0, n_blocks, size=n_blocks)
    return blocks[pick].reshape(-1)

def block_bootstrap_stat(x, block_size, n_samples=1000, estimator=np.mean):
    vals = np.zeros(n_samples)
    for i in range(n_samples):
        idx = block_bootstrap_indices(len(x), block_size)
        vals[i] = estimator(x[idx])
    return np.mean(vals), np.std(vals, ddof=1), vals

def block_jackknife_stat(x, block_size, estimator=np.mean):
    blocks, trimmed = _make_blocks(len(x), block_size)
    if blocks is None:
        full = estimator(x)
        return full, np.nan, np.nan

    x_trim = x[:trimmed]
    n_blocks = blocks.shape[0]
    jk_vals = np.zeros(n_blocks)

    for i in range(n_blocks):
        mask = np.ones(trimmed, dtype=bool)
        mask[i * block_size:(i + 1) * block_size] = False
        jk_vals[i] = estimator(x_trim[mask])

    mean_jk = np.mean(jk_vals)
    err_jk = np.sqrt((n_blocks - 1) * np.var(jk_vals, ddof=0))
    full_est = estimator(x_trim)
    bias = (n_blocks - 1) * (mean_jk - full_est)
    return mean_jk, err_jk, bias

block_size = max(1, int(np.ceil(2.0 * tau_int)))
if n_meas // block_size < 4:
    block_size = max(1, n_meas // 4)

print(f"\n--- BLOCK RESAMPLING SETUP ---")
print(f"Block size: {block_size} measurements")
print(f"Number of blocks: {max(1, n_meas // max(1, block_size))}")

resamp_mean, resamp_err, _ = block_bootstrap_stat(
    plaq_timeseries,
    block_size,
    n_samples=PLAQ_BOOTSTRAP_SAMPLES,
)
print(f"\n--- BLOCK BOOTSTRAP ANALYSIS ---")
print(f"Mean plaquette: {resamp_mean:.6f} +/- {resamp_err:.6f}")

jk_mean, jk_err, jk_bias = block_jackknife_stat(plaq_timeseries, block_size)
print(f"\n--- BLOCK JACKKNIFE ANALYSIS ---")
print(f"Mean plaquette: {jk_mean:.6f} +/- {jk_err:.6f}")
print(f"Estimated bias: {jk_bias:.2e}")

# =============================================================================
# 4. GEVP MASS WITH BLOCK BOOTSTRAP
# =============================================================================

def build_correlator_matrix(ops_history):
    """Build the GEVP correlation matrix C(t)."""
    n_meas_loc, n_ops_loc, Nt_loc = ops_history.shape

    # Subtract VEV
    vevs = np.mean(ops_history, axis=(0, 2))
    ops_sub = ops_history - vevs[None, :, None]

    # Build C(t)
    C_matrix = np.zeros((Nt_loc, n_ops_loc, n_ops_loc))
    for t in range(Nt_loc):
        for i in range(n_ops_loc):
            for j in range(n_ops_loc):
                prod = ops_sub[:, i, :] * np.roll(ops_sub[:, j, :], -t, axis=1)
                C_matrix[t, i, j] = np.mean(prod)

    # Symmetrize
    C_matrix = 0.5 * (C_matrix + np.transpose(C_matrix, (0, 2, 1)))

    # Fold
    for t in range(1, Nt_loc // 2 + 1):
        C_matrix[t] = 0.5 * (C_matrix[t] + C_matrix[Nt_loc - t])

    return C_matrix[: Nt_loc // 2]

def extract_mass_from_correlator(C_matrix):
    """Extract ground-state effective mass from GEVP eigenvalue ratio."""
    Nt_half, n_ops_loc, _ = C_matrix.shape

    eig_vals = np.zeros((Nt_half, n_ops_loc))
    t0 = GEVP_T0
    if Nt_half <= t0:
        return np.nan

    for t in range(Nt_half):
        try:
            evals = la.eigh(C_matrix[t], C_matrix[t0], eigvals_only=True)
            eig_vals[t, :] = np.sort(evals)[::-1]
        except (np.linalg.LinAlgError, ValueError):
            eig_vals[t, :] = np.nan

    lambda_0 = eig_vals[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        m_eff = np.log(lambda_0[:-1] / lambda_0[1:])

    if len(m_eff) > 2 and np.isfinite(m_eff[2]):
        return m_eff[2]
    return np.nan

def extract_string_tension(wilson_mean):
    """Extract string tension from Wilson loop data."""
    R_max = wilson_mean.shape[0]
    V_R = np.full(R_max, np.nan)

    # Extract V(R) from Wilson loop ratios at T=3,4
    for r in range(R_max):
        if wilson_mean[r, 3] > 0 and wilson_mean[r, 4] > 0:
            V_R[r] = np.log(wilson_mean[r, 3] / wilson_mean[r, 4])

    # Fit V(R) = sigma * R + const for R >= 2
    r_vals = np.arange(1, R_max + 1)
    mask = np.isfinite(V_R) & (r_vals >= 2)
    if np.sum(mask) >= 2:
        try:
            popt, _ = curve_fit(lambda r, s, c: s * r + c, r_vals[mask], V_R[mask])
            return popt[0]
        except (RuntimeError, ValueError):
            return np.nan
    return np.nan

def block_bootstrap_physics(ops_history, wilson_history, block_size, n_samples=200):
    mass_samples = []
    sigma_samples = []
    ratio_samples = []
    n_cfg = ops_history.shape[0]
    progress_step = max(1, n_samples // 10)

    for i in range(n_samples):
        idx = block_bootstrap_indices(n_cfg, block_size)
        ops_resamp = ops_history[idx]

        mass = extract_mass_from_correlator(build_correlator_matrix(ops_resamp))
        if np.isfinite(mass):
            mass_samples.append(mass)

        sigma = np.nan
        if wilson_history is not None:
            wilson_mean = np.mean(wilson_history[idx], axis=0)
            sigma = extract_string_tension(wilson_mean)
            if np.isfinite(sigma):
                sigma_samples.append(sigma)

        if np.isfinite(mass) and np.isfinite(sigma) and sigma > 0:
            ratio_samples.append(mass / np.sqrt(sigma))

        if (i + 1) % progress_step == 0 or i == 0 or (i + 1) == n_samples:
            print(f"  bootstrap progress: {i + 1}/{n_samples}", flush=True)

    return np.array(mass_samples), np.array(sigma_samples), np.array(ratio_samples)

print(f"\n--- GEVP MASS + STRING TENSION WITH BLOCK BOOTSTRAP ---")
print("Running block bootstrap (this may take a moment)...")

# Optional downsampling only for the expensive physics bootstrap.
ops_phys = ops_history[::PHYSICS_BOOTSTRAP_STRIDE]
wilson_phys = wilson_history[::PHYSICS_BOOTSTRAP_STRIDE] if wilson_history is not None else None
if PHYSICS_BOOTSTRAP_CFG_LIMIT > 0 and ops_phys.shape[0] > PHYSICS_BOOTSTRAP_CFG_LIMIT:
    step = int(np.ceil(ops_phys.shape[0] / PHYSICS_BOOTSTRAP_CFG_LIMIT))
    ops_phys = ops_phys[::step]
    wilson_phys = wilson_phys[::step] if wilson_phys is not None else None

# Keep at least ~4 blocks after any downsampling.
phys_block_size = max(1, int(np.ceil(block_size / PHYSICS_BOOTSTRAP_STRIDE)))
if ops_phys.shape[0] // phys_block_size < 4:
    phys_block_size = max(1, ops_phys.shape[0] // 4)

print(f"Physics bootstrap configs used: {ops_phys.shape[0]}")
print(f"Physics bootstrap block size: {phys_block_size}")
print(f"Physics bootstrap samples: {PHYSICS_BOOTSTRAP_SAMPLES}")

mass_samples, sigma_samples, ratio_samples = block_bootstrap_physics(
    ops_phys,
    wilson_phys,
    phys_block_size,
    n_samples=PHYSICS_BOOTSTRAP_SAMPLES,
)

mass_resamp = np.mean(mass_samples) if len(mass_samples) > 10 else np.nan
mass_err_resamp = np.std(mass_samples, ddof=1) if len(mass_samples) > 10 else np.nan
print(f"Glueball mass: {mass_resamp:.4f} +/- {mass_err_resamp:.4f}")

# =============================================================================
# 5. STRING TENSION AND DIMENSIONLESS RATIO
# =============================================================================

sigma = extract_string_tension(wilson_avg)
print(f"\n--- STRING TENSION ---")
print(f"String tension (sigma*a^2): {sigma:.4f}")

if len(sigma_samples) > 10:
    sigma_err = np.std(sigma_samples, ddof=1)
else:
    sigma_err = np.nan

if len(ratio_samples) > 10:
    ratio = np.mean(ratio_samples)
    ratio_err = np.std(ratio_samples, ddof=1)
    print(f"\n--- DIMENSIONLESS RATIO ---")
    print(f"m_G / sqrt(sigma): {ratio:.4f} +/- {ratio_err:.4f}")
    print(f"(Expected for 3D SU(2) 0++ glueball: ~4-5)")
elif np.isfinite(sigma) and sigma > 0 and np.isfinite(mass_resamp):
    ratio = mass_resamp / np.sqrt(sigma)
    ratio_err = mass_err_resamp / np.sqrt(sigma)
    print(f"\n--- DIMENSIONLESS RATIO ---")
    print(f"m_G / sqrt(sigma): {ratio:.4f} +/- {ratio_err:.4f} (sigma error unavailable)")
    print(f"(Expected for 3D SU(2) 0++ glueball: ~4-5)")
else:
    ratio = np.nan
    ratio_err = np.nan

# =============================================================================
# 6. PLOTTING
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Autocorrelation function
ax1 = axes[0, 0]
t_acf = np.arange(len(acf))
ax1.plot(t_acf, acf, 'b-', lw=1)
ax1.axhline(0, color='k', ls='--', lw=0.5)
ax1.axvline(window, color='r', ls='--', label=f'Window={window}')
ax1.axhline(0.1, color='gray', ls=':', lw=0.5)
ax1.set_xlabel('Lag (sweeps)')
ax1.set_ylabel('Autocorrelation C(t)')
ax1.set_title(f'Autocorrelation Function (tau_int={tau_int:.1f})')
ax1.legend()
ax1.set_xlim(0, min(100, len(acf)))
ax1.grid(True, alpha=0.3)

# Plot 2: Binning analysis
ax2 = axes[0, 1]
ax2.plot(bin_sizes, bin_errors, 'bo-', markersize=3)
ax2.axhline(naive_error, color='r', ls='--', label='Naive error')
ax2.axhline(bin_errors[-1], color='g', ls='--', label='Plateau error')
ax2.set_xlabel('Bin size')
ax2.set_ylabel('Error estimate')
ax2.set_title('Binning Analysis')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Time series with running mean
ax3 = axes[1, 0]
window_size = min(50, max(1, n_meas))
running_mean = np.convolve(plaq_timeseries, np.ones(window_size)/window_size, mode='valid')
ax3.plot(plaq_timeseries, 'b-', alpha=0.3, lw=0.5, label='Raw data')
ax3.plot(np.arange(window_size-1, n_meas), running_mean, 'r-', lw=1, label=f'Running mean (w={window_size})')
ax3.axhline(np.mean(plaq_timeseries), color='k', ls='--', lw=1)
ax3.set_xlabel('Configuration')
ax3.set_ylabel('Plaquette sum')
ax3.set_title('Monte Carlo Time Series')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error comparison
ax4 = axes[1, 1]
methods = ['Naive', 'Binned', 'Block BS', 'Block JK']
errors = [naive_error, bin_errors[-1], resamp_err, jk_err]
colors = ['blue', 'green', 'orange', 'red']
bars = ax4.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Error estimate')
ax4.set_title('Error Estimation Comparison')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, err in zip(bars, errors):
    label = f'{err:.5f}' if np.isfinite(err) else 'nan'
    y_text = bar.get_height() + 0.0001 if np.isfinite(bar.get_height()) else 0.0
    ax4.text(bar.get_x() + bar.get_width()/2, y_text,
             label, ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save figure
output_path = os.path.join(script_dir, 'latticeYM3d_errors.png')
plt.savefig(output_path, dpi=150)
print(f"\nFigure saved to: {output_path}")
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY OF ERROR ANALYSIS")
print("="*60)
print(f"Number of configurations: {n_meas}")
print(f"Integrated autocorrelation time: {tau_int:.2f}")
print(f"Effective independent samples: {n_meas / (2*tau_int):.1f}")
print(f"\nPlaquette mean: {np.mean(plaq_timeseries):.6f}")
print(f"  Naive error:     {naive_error:.6f}")
print(f"  Binned error:    {bin_errors[-1]:.6f}")
print(f"  Block BS error:  {resamp_err:.6f}")
print(f"  Block JK error:  {jk_err:.6f}")
print(f"\nGlueball mass (block BS): {mass_resamp:.4f} +/- {mass_err_resamp:.4f}")
if np.isfinite(sigma_err):
    print(f"String tension (sigma*a^2): {sigma:.4f} +/- {sigma_err:.4f}")
else:
    print(f"String tension (sigma*a^2): {sigma:.4f}")
if not np.isnan(ratio):
    print(f"m_G / sqrt(sigma): {ratio:.4f} +/- {ratio_err:.4f}")
print("="*60)
