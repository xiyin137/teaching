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
#   3. Resampling (Efron's bootstrap) - non-parametric error estimation
#   4. Jackknife resampling - bias-corrected error estimation
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
wilson_avg = data['wilson_avg']
L = int(data['L'])
beta = float(data['beta'])

n_meas, n_ops, Nt = ops_history.shape
print(f"Loaded Data: Beta={beta}, L={L}, Configs={n_meas}")

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
# 3. RESAMPLING ERROR ESTIMATION (Efron's bootstrap)
# =============================================================================
# Note: This is statistical resampling, not to be confused with the
# conformal bootstrap which uses crossing symmetry constraints in CFTs.
#
# CAVEAT FOR CORRELATED DATA:
# Standard resampling assumes independent samples. For correlated Monte Carlo
# data, it can underestimate errors (similar to the naive error). Solutions:
#   1. Use block resampling (resample blocks of consecutive measurements)
#   2. Thin the data first (use only every tau_int-th measurement)
#   3. Compare with binning analysis to verify consistency
# Here we use standard resampling but rely on n_skip being large enough
# that measurements are approximately independent.

def resampling_error(x, n_samples=1000, estimator=np.mean):
    """
    Resampling method for error estimation (Efron 1979).

    Resample with replacement N times, compute statistic on each sample.
    Error is the standard deviation of the resampled distribution.

    Works for any estimator (mean, median, fitted parameters, etc.)
    """
    n = len(x)
    resampled_values = np.zeros(n_samples)

    for i in range(n_samples):
        # Resample with replacement
        idx = np.random.randint(0, n, size=n)
        resampled_values[i] = estimator(x[idx])

    return np.mean(resampled_values), np.std(resampled_values, ddof=1)

# Resampling estimate of the mean plaquette
resamp_mean, resamp_err = resampling_error(plaq_timeseries, n_samples=1000)
print(f"\n--- RESAMPLING ANALYSIS ---")
print(f"Mean plaquette: {resamp_mean:.6f} +/- {resamp_err:.6f}")

# =============================================================================
# 4. JACKKNIFE ERROR ESTIMATION
# =============================================================================

def jackknife_error(x, estimator=np.mean):
    """
    Jackknife resampling for error estimation.

    Leave-one-out: compute statistic on each subset with one sample removed.
    Error is sqrt((n-1) * variance of jackknife values).

    Jackknife is particularly good for bias estimation and correction.
    """
    n = len(x)
    jackknife_values = np.zeros(n)

    for i in range(n):
        # Leave out sample i
        subset = np.concatenate([x[:i], x[i+1:]])
        jackknife_values[i] = estimator(subset)

    mean_jk = np.mean(jackknife_values)
    # Jackknife error formula
    err_jk = np.sqrt((n-1) * np.var(jackknife_values, ddof=0))

    # Bias estimation
    full_estimate = estimator(x)
    bias = (n-1) * (mean_jk - full_estimate)

    return mean_jk, err_jk, bias

jk_mean, jk_err, jk_bias = jackknife_error(plaq_timeseries)
print(f"\n--- JACKKNIFE ANALYSIS ---")
print(f"Mean plaquette: {jk_mean:.6f} +/- {jk_err:.6f}")
print(f"Estimated bias: {jk_bias:.2e}")

# =============================================================================
# 5. GEVP WITH RESAMPLING ERRORS
# =============================================================================
#
# For complex derived quantities like the glueball mass (which involves
# building a correlation matrix, solving a generalized eigenvalue problem,
# and extracting a mass from the eigenvalue decay), analytic error propagation
# is impractical. Resampling provides a straightforward solution:
#
#   1. Resample the raw configurations with replacement
#   2. For each resampled ensemble, compute the full analysis pipeline
#   3. The spread of results gives the statistical error
#
# This automatically accounts for all correlations in the analysis chain.

def build_correlator_matrix(ops_history):
    """Build the GEVP correlation matrix C(t)."""
    n_meas, n_ops, Nt = ops_history.shape

    # Subtract VEV
    vevs = np.mean(ops_history, axis=(0, 2))
    ops_sub = ops_history - vevs[None, :, None]

    # Build C(t)
    C_matrix = np.zeros((Nt, n_ops, n_ops))
    for t in range(Nt):
        for i in range(n_ops):
            for j in range(n_ops):
                prod = ops_sub[:, i, :] * np.roll(ops_sub[:, j, :], -t, axis=1)
                C_matrix[t, i, j] = np.mean(prod)

    # Symmetrize
    C_matrix = 0.5 * (C_matrix + np.transpose(C_matrix, (0, 2, 1)))

    # Fold
    for t in range(1, Nt//2 + 1):
        C_matrix[t] = 0.5 * (C_matrix[t] + C_matrix[Nt-t])

    return C_matrix[:Nt//2]

def extract_mass_from_correlator(C_matrix):
    """Extract ground state mass from GEVP."""
    Nt_half, n_ops, _ = C_matrix.shape

    # Solve GEVP
    eig_vals = np.zeros((Nt_half, n_ops))
    t0 = 1  # Use t0=1 to avoid contact terms

    for t in range(Nt_half):
        try:
            evals = la.eigh(C_matrix[t], C_matrix[t0], eigvals_only=True)
            eig_vals[t, :] = np.sort(evals)[::-1]
        except:
            eig_vals[t, :] = np.nan

    lambda_0 = eig_vals[:, 0]

    # Effective mass from ratio: m_eff(t) = log(C(t)/C(t+1))
    with np.errstate(divide='ignore', invalid='ignore'):
        m_eff = np.log(lambda_0[:-1] / lambda_0[1:])

    # Return effective mass at t=2 (after contact term effects)
    if len(m_eff) > 2 and np.isfinite(m_eff[2]):
        return m_eff[2]
    return np.nan

def resampling_gevp_mass(ops_history, n_samples=200):
    """Resampling error for GEVP mass extraction."""
    n_meas = ops_history.shape[0]
    mass_samples = []

    for _ in range(n_samples):
        # Resample configurations
        idx = np.random.randint(0, n_meas, size=n_meas)
        ops_resamp = ops_history[idx]

        # Build correlator and extract mass
        C = build_correlator_matrix(ops_resamp)
        m = extract_mass_from_correlator(C)
        if np.isfinite(m):
            mass_samples.append(m)

    if len(mass_samples) > 10:
        return np.mean(mass_samples), np.std(mass_samples, ddof=1)
    return np.nan, np.nan

print(f"\n--- GEVP MASS WITH RESAMPLING ERRORS ---")
print("Running resampling (this may take a moment)...")
mass_resamp, mass_err_resamp = resampling_gevp_mass(ops_history, n_samples=200)
print(f"Glueball mass: {mass_resamp:.4f} +/- {mass_err_resamp:.4f}")

# =============================================================================
# 6. STRING TENSION AND DIMENSIONLESS RATIO
# =============================================================================
#
# The string tension sigma sets the characteristic scale of the confining theory.
# It can be extracted from Wilson loops: W(R,T) ~ exp(-V(R)*T) where
# V(R) = sigma*R + const for large R (linear confinement).
#
# The dimensionless ratio m_G/sqrt(sigma) is a universal prediction:
# - Independent of the lattice spacing (both m_G and sqrt(sigma) have dim [1/length])
# - Can be compared to continuum extrapolations and other methods
# - For 3D SU(2) Yang-Mills 0++ glueball: m_G/sqrt(sigma) ~ 4-5
#
# NOTE: We extract sigma from wilson_avg which is already averaged over all
# configurations. A proper resampling analysis of sigma would require storing
# Wilson loops per-configuration, which we don't do here for simplicity.

def extract_string_tension(wilson_avg):
    """Extract string tension from Wilson loop data."""
    R_max = wilson_avg.shape[0]
    V_R = np.zeros(R_max)
    V_R[:] = np.nan

    # Extract V(R) from Wilson loop ratios at T=3,4
    for r in range(R_max):
        if wilson_avg[r, 3] > 0 and wilson_avg[r, 4] > 0:
            V_R[r] = np.log(wilson_avg[r, 3] / wilson_avg[r, 4])

    # Fit V(R) = sigma * R + const for R >= 2
    r_vals = np.arange(1, R_max+1)
    mask = np.isfinite(V_R) & (r_vals >= 2)
    if np.sum(mask) >= 2:
        try:
            popt, _ = curve_fit(lambda r, s, c: s*r + c, r_vals[mask], V_R[mask])
            return popt[0]  # sigma
        except:
            pass
    return np.nan

# Get string tension from full data
sigma = extract_string_tension(wilson_avg)
print(f"\n--- STRING TENSION ---")
print(f"String tension (sigma*a^2): {sigma:.4f}")

# Compute dimensionless ratio m_G / sqrt(sigma)
if not np.isnan(sigma) and sigma > 0 and not np.isnan(mass_resamp):
    ratio = mass_resamp / np.sqrt(sigma)
    # Error propagation: d(m/sqrt(s)) = dm/sqrt(s) (ignoring sigma error for now)
    ratio_err = mass_err_resamp / np.sqrt(sigma)
    print(f"\n--- DIMENSIONLESS RATIO ---")
    print(f"m_G / sqrt(sigma): {ratio:.4f} +/- {ratio_err:.4f}")
    print(f"(Expected for 3D SU(2) 0++ glueball: ~4-5)")
else:
    ratio = np.nan
    ratio_err = np.nan

# =============================================================================
# 7. PLOTTING
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
window_size = 50
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
methods = ['Naive', 'Binned', 'Resampling', 'Jackknife']
errors = [naive_error, bin_errors[-1], resamp_err, jk_err]
colors = ['blue', 'green', 'orange', 'red']
bars = ax4.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Error estimate')
ax4.set_title('Error Estimation Comparison')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, err in zip(bars, errors):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{err:.5f}', ha='center', va='bottom', fontsize=9)

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
print(f"  Resampling error: {resamp_err:.6f}")
print(f"  Jackknife error: {jk_err:.6f}")
print(f"\nGlueball mass (resampling): {mass_resamp:.4f} +/- {mass_err_resamp:.4f}")
print(f"String tension (sigma*a^2): {sigma:.4f}")
if not np.isnan(ratio):
    print(f"m_G / sqrt(sigma): {ratio:.4f} +/- {ratio_err:.4f}")
print("="*60)
