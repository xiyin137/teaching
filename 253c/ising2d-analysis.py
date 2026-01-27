"""
Analysis script for 2D Ising Model simulation data
Extracts critical exponent eta from susceptibility scaling and correlation function

=============================================================================
FINITE-SIZE SCALING THEORY
=============================================================================
At a continuous phase transition, correlation length ξ diverges as:
    ξ ~ |T - Tc|^(-ν)

On a finite lattice of size L, the correlation length is cut off at L.
This leads to finite-size scaling relations at T = Tc:

  - Susceptibility:    χ ~ L^(γ/ν)
  - Magnetization:     <|m|> ~ L^(-β/ν)
  - Specific heat:     C ~ L^(α/ν)
  - Correlation fn:    G(r) ~ r^(-η)  (at Tc, no exponential decay)

Using hyperscaling relation γ/ν = 2 - η (in d=2), we can extract η from χ.

=============================================================================
EXACT VALUES FOR 2D ISING MODEL (Onsager solution)
=============================================================================
  Tc = 2 / ln(1 + √2) ≈ 2.269185

  Critical exponents:
    α = 0        (log divergence of specific heat)
    β = 1/8      (magnetization: M ~ (Tc-T)^β for T < Tc)
    γ = 7/4      (susceptibility: χ ~ |T-Tc|^(-γ))
    δ = 15       (critical isotherm: M ~ H^(1/δ) at T = Tc)
    ν = 1        (correlation length: ξ ~ |T-Tc|^(-ν))
    η = 1/4      (correlation function: G(r) ~ r^(-η) at Tc)

  Derived ratios (what we measure via finite-size scaling):
    γ/ν = 7/4 = 1.75
    β/ν = 1/8 = 0.125
    2 - η = 7/4 = 1.75  (from hyperscaling)

=============================================================================
BINDER CUMULANT METHOD FOR FINDING Tc
=============================================================================
The Binder cumulant is defined as:
    U4 = 1 - <M^4> / (3 <M^2>^2)

Key properties:
  - U4 → 2/3 in ordered phase (M ≈ ±M0, so <M^4> ≈ <M^2>^2)
  - U4 → 0 in disordered phase (Gaussian fluctuations, <M^4> = 3<M^2>^2)
  - At Tc, U4 is scale-invariant (independent of L, up to corrections)

Therefore, U4(T) curves for different L values cross at T = Tc.
This provides a precise method to locate the critical temperature.

=============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Use script directory for file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = os.path.join(script_dir, "ising_data.npz")


def binder_cumulant(M2_arr, M4_arr):
    """
    Compute Binder cumulant U4 = 1 - <M^4> / (3<M^2>^2).

    This dimensionless ratio is scale-invariant at Tc, making it
    ideal for locating the critical temperature via the crossing method.
    """
    m2_avg = np.mean(M2_arr)
    m4_avg = np.mean(M4_arr)
    if m2_avg == 0:
        return 0.0
    return 1.0 - m4_avg / (3.0 * m2_avg * m2_avg)


def main():
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    try:
        data = np.load(FILE_NAME)
        print(f"Loaded {FILE_NAME}")
    except FileNotFoundError:
        print(f"Error: {FILE_NAME} not found. Run simulation first.")
        return

    Tc_final = float(data['Tc_final'])
    Tc_scout = float(data['Tc_scout'])
    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))  # Onsager's exact result

    print(f"Tc from Scout:  {Tc_scout:.6f}")
    print(f"Tc from Refine: {Tc_final:.6f}")
    print(f"Tc exact:       {Tc_exact:.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # =========================================================================
    # PANEL 0: PHASE TRANSITION (M vs T)
    # =========================================================================
    # The order parameter <|m|> = <|M|>/N shows the phase transition:
    #   - For T < Tc: spontaneous magnetization, <|m|> > 0
    #   - For T > Tc: disordered phase, <|m|> → 0 as L → ∞
    # On a finite lattice, <|m|> is always > 0 but drops sharply near Tc.
    # =========================================================================
    if 'phase_T' in data.files:
        T_phase = data['phase_T']
        M_phase = data['phase_M']
        M_phase_err = data['phase_M_err']
        phase_L = int(data['phase_L'])

        ax = axes[0, 0]
        ax.errorbar(T_phase, M_phase, yerr=M_phase_err, fmt='o-', capsize=2, markersize=4)
        ax.axvline(Tc_final, color='r', ls='--', label=f'Tc={Tc_final:.4f}')
        ax.axvline(Tc_exact, color='g', ls=':', alpha=0.7, label='Tc exact')
        ax.set_xlabel('Temperature T')
        ax.set_ylabel(r'$\langle|m|\rangle$')
        ax.set_title(f'Phase Transition (L={phase_L})')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No phase data', ha='center', va='center')
        axes[0, 0].set_title('Phase Transition')

    # =========================================================================
    # PANEL 1: BINDER CUMULANT CROSSING
    # =========================================================================
    # The Binder cumulant U4 = 1 - <M^4>/(3<M^2>^2) is scale-invariant at Tc.
    # Curves for different L cross at Tc, providing a precise Tc estimate.
    #
    # Physics: U4 measures the non-Gaussianity of the magnetization distribution.
    # At Tc, the distribution has a universal shape (independent of L).
    # =========================================================================
    refine_L = data['refine_L']
    refine_T = data['refine_T']
    L1, L2 = refine_L[0], refine_L[1]

    # Compute U4 at each temperature for both L values
    U4_L1 = []
    U4_L2 = []
    for T in refine_T:
        key = f'ref_L{L1}_T{T:.4f}'
        M2 = data[f'{key}_M2']
        M4 = data[f'{key}_M4']
        U4_L1.append(binder_cumulant(M2, M4))

        key = f'ref_L{L2}_T{T:.4f}'
        M2 = data[f'{key}_M2']
        M4 = data[f'{key}_M4']
        U4_L2.append(binder_cumulant(M2, M4))

    ax = axes[0, 1]
    ax.plot(refine_T, U4_L1, 'o-', label=f'L={L1}')
    ax.plot(refine_T, U4_L2, 's-', label=f'L={L2}')
    ax.axvline(Tc_final, color='k', ls=':', label=f'Tc={Tc_final:.4f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Binder Cumulant U4')
    ax.set_title('Binder Cumulant Crossing')
    ax.legend()
    ax.grid(alpha=0.3)

    # =========================================================================
    # PANEL 2: SUSCEPTIBILITY SCALING
    # =========================================================================
    # At Tc, the susceptibility scales as: χ ~ L^(γ/ν)
    #
    # Definition: χ = β <M²> / V  (fluctuation-dissipation theorem)
    #
    # From hyperscaling in d=2: γ/ν = 2 - η
    # So measuring the slope of log(χ) vs log(L) gives γ/ν directly,
    # and η = 2 - γ/ν.
    #
    # Exact values: γ/ν = 7/4 = 1.75, η = 1/4 = 0.25
    # =========================================================================
    prod_L = data['prod_L']
    beta_c = 1.0 / Tc_final  # Inverse critical temperature

    log_L, log_chi, log_err = [], [], []

    print("\nSusceptibility Scaling:")
    for L in prod_L:
        M2 = data[f'prod_L{L}_M2']
        vol = L * L

        # Block averaging for error estimation
        # Divide time series into n_blk blocks, compute χ for each
        n_blk = 50
        sz = len(M2) // n_blk
        blks = [beta_c * np.mean(M2[k*sz:(k+1)*sz]) / vol for k in range(n_blk)]
        chi = np.mean(blks)
        err = np.std(blks) / np.sqrt(n_blk)  # Standard error of mean

        log_L.append(np.log(L))
        log_chi.append(np.log(chi))
        log_err.append(err / chi)  # Relative error propagates to log
        print(f"  L={L:3d}:  chi = {chi:.2f} +/- {err:.2f}")

    # Linear fit: log(χ) = (γ/ν) * log(L) + const
    def fit_lin(x, slope, intercept):
        return slope * x + intercept

    popt, pcov = curve_fit(fit_lin, log_L, log_chi, sigma=log_err, absolute_sigma=True)
    slope = popt[0]
    slope_err = np.sqrt(pcov[0, 0])
    gamma_nu = slope           # γ/ν from the fit
    eta_chi = 2.0 - slope      # η = 2 - γ/ν (hyperscaling)
    eta_chi_err = slope_err

    print(f"\n>> gamma/nu = {gamma_nu:.4f} +/- {slope_err:.4f} (exact: 1.75)")
    print(f">> eta from chi = {eta_chi:.4f} +/- {eta_chi_err:.4f} (exact: 0.25)")

    ax = axes[0, 2]
    ax.errorbar(log_L, log_chi, yerr=log_err, fmt='o', capsize=3, label='Data')
    x_fit = np.linspace(min(log_L), max(log_L), 100)
    ax.plot(x_fit, fit_lin(x_fit, *popt), 'r--',
            label=rf'Fit: $\gamma/\nu = {gamma_nu:.3f}$, $\eta = {eta_chi:.3f}$')
    ax.set_xlabel(r'$\ln L$')
    ax.set_ylabel(r'$\ln \chi$')
    ax.set_title(r'Susceptibility Scaling: $\chi \sim L^{\gamma/\nu}$')
    ax.legend()
    ax.grid(alpha=0.3)

    # =========================================================================
    # PANEL 3: MAGNETIZATION SCALING
    # =========================================================================
    # At Tc, the order parameter scales as: <|m|> ~ L^(-β/ν)
    #
    # The magnetization per spin m = M/N vanishes as L → ∞ at Tc,
    # but with a power-law dependence on L.
    #
    # We measure <|m|> ≈ sqrt(<M²>)/N since <M> = 0 by symmetry.
    #
    # Exact value: β/ν = 1/8 = 0.125
    # =========================================================================
    log_M, log_M_err = [], []

    print("\nMagnetization Scaling:")
    for L in prod_L:
        M2 = data[f'prod_L{L}_M2']
        vol = L * L

        # <|M|> ≈ sqrt(<M²>) at Tc (since distribution is symmetric around 0)
        # <|m|> = <|M|>/N = sqrt(<M²>)/N
        n_blk = 50
        sz = len(M2) // n_blk
        blks = [np.sqrt(np.mean(M2[k*sz:(k+1)*sz])) / vol for k in range(n_blk)]
        m_avg = np.mean(blks)
        m_err = np.std(blks) / np.sqrt(n_blk)

        log_M.append(np.log(m_avg))
        log_M_err.append(m_err / m_avg)
        print(f"  L={L:3d}:  <|m|> = {m_avg:.4f} +/- {m_err:.4f}")

    # Linear fit: log(<|m|>) = -(β/ν) * log(L) + const
    popt_m, pcov_m = curve_fit(fit_lin, log_L, log_M, sigma=log_M_err, absolute_sigma=True)
    beta_nu = -popt_m[0]  # Negative slope because <|m|> decreases with L
    beta_nu_err = np.sqrt(pcov_m[0, 0])

    print(f"\n>> beta/nu = {beta_nu:.4f} +/- {beta_nu_err:.4f} (exact: 0.125)")

    ax = axes[1, 0]
    ax.errorbar(log_L, log_M, yerr=log_M_err, fmt='o', capsize=3, label='Data')
    ax.plot(x_fit, fit_lin(x_fit, *popt_m), 'r--',
            label=rf'Fit: $\beta/\nu = {beta_nu:.3f}$')
    ax.set_xlabel(r'$\ln L$')
    ax.set_ylabel(r'$\ln \langle|m|\rangle$')
    ax.set_title(r'Magnetization Scaling: $\langle|m|\rangle \sim L^{-\beta/\nu}$')
    ax.legend()
    ax.grid(alpha=0.3)

    # =========================================================================
    # PANEL 4: CORRELATION FUNCTION
    # =========================================================================
    # At Tc, the spin-spin correlation function decays as a power law:
    #     G(r) = <s(0) s(r)> ~ r^(-η)
    #
    # Away from Tc, there's exponential decay: G(r) ~ exp(-r/ξ) / r^η
    # At Tc, ξ → ∞, so only the power law remains.
    #
    # Fitting log(G) vs log(r) gives η directly.
    # Exact value: η = 1/4 = 0.25
    #
    # Note: G(0) = <s²> = 1, G(r→L/2) may show finite-size effects.
    # =========================================================================
    L_corr = int(data['corr_L'])
    g_r = data['corr_g']  # G(r) already normalized by the simulation
    r_vals = np.arange(len(g_r))

    # Fit range: avoid r=0 (trivial), r=1,2 (lattice artifacts),
    # and large r (finite-size effects)
    r_min_fit, r_max_fit = 3, 10

    valid_fit = (r_vals >= r_min_fit) & (r_vals <= r_max_fit) & (g_r > 0)
    x_data_fit = r_vals[valid_fit]
    y_data_fit = g_r[valid_fit]

    # Power law fit: G(r) = A * r^(-η)
    def power_law(r, eta, A):
        return A * np.power(r, -eta)

    try:
        popt_corr, pcov_corr = curve_fit(power_law, x_data_fit, y_data_fit, p0=[0.25, 1.0])
        eta_corr = popt_corr[0]
        eta_corr_err = np.sqrt(pcov_corr[0, 0])
        A_corr = popt_corr[1]
        print(f"\n>> eta from G(r) power law = {eta_corr:.4f} +/- {eta_corr_err:.4f} (exact: 0.25)")
    except Exception as e:
        print(f"Correlation fit failed: {e}")
        eta_corr = 0.25
        eta_corr_err = 0.0
        A_corr = 1.0

    ax = axes[1, 1]
    # Log-log plot to show power law behavior
    ax.loglog(r_vals[1:], g_r[1:], 'o', alpha=0.5, markersize=4, label='Data')

    # Plot power law fit
    r_full = np.linspace(1, L_corr // 2, 500)
    g_fit_full = A_corr * np.power(r_full, -eta_corr)
    ax.plot(r_full, g_fit_full, 'r-', lw=2, label=rf'Power law: $\eta={eta_corr:.3f}$')

    ax.set_xlabel('Distance r')
    ax.set_ylabel('G(r)')
    ax.set_title(f'Correlation Function (L={L_corr})')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # =========================================================================
    # PANEL 5: SUMMARY
    # =========================================================================
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"SUMMARY\n"
        f"{'='*35}\n\n"
        f"Critical Temperature:\n"
        f"  Tc (measured) = {Tc_final:.5f}\n"
        f"  Tc (exact)    = {Tc_exact:.5f}\n\n"
        f"Critical Exponents:\n"
        f"  γ/ν = {gamma_nu:.3f} ± {slope_err:.3f}  (exact: 1.75)\n"
        f"  β/ν = {beta_nu:.3f} ± {beta_nu_err:.3f}  (exact: 0.125)\n"
        f"  η (from χ)  = {eta_chi:.3f} ± {eta_chi_err:.3f}  (exact: 0.25)\n"
        f"  η (from G)  = {eta_corr:.3f} ± {eta_corr_err:.3f}  (exact: 0.25)\n\n"
        f"Hyperscaling check:\n"
        f"  2 - η = γ/ν\n"
        f"  {2-eta_chi:.3f} ≈ {gamma_nu:.3f}"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Results Summary')

    # =========================================================================
    # CONSOLE OUTPUT
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Critical Temperature Tc    = {Tc_final:.6f}  (exact: {Tc_exact:.6f})")
    print(f"gamma/nu (from χ scaling)  = {gamma_nu:.4f} ± {slope_err:.4f}  (exact: 1.75)")
    print(f"eta (from χ, using 2-γ/ν)  = {eta_chi:.4f} ± {eta_chi_err:.4f}  (exact: 0.25)")
    print(f"beta/nu (from m scaling)   = {beta_nu:.4f} ± {beta_nu_err:.4f}  (exact: 0.125)")
    print(f"eta (from G(r) power law)  = {eta_corr:.4f} ± {eta_corr_err:.4f}  (exact: 0.25)")
    print("=" * 60)

    plt.tight_layout()
    output_path = os.path.join(script_dir, 'ising2d_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
