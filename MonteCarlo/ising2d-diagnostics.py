"""
Diagnostics for 2D Ising simulation data
- Time series plots (check for drift/thermalization)
- Autocorrelation analysis (measure τ_int)
- Binning analysis for error estimation

Run this after simulation to verify data quality.
"""
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = "ising_data.npz"


def autocorrelation(x, max_lag=1000):
    """Compute normalized autocorrelation function."""
    x = x - np.mean(x)
    n = len(x)
    max_lag = min(max_lag, n // 4)
    acf = np.zeros(max_lag)
    var = np.var(x)
    if var == 0:
        return acf
    for lag in range(max_lag):
        acf[lag] = np.mean(x[:n-lag] * x[lag:]) / var
    return acf


def integrated_autocorr_time(acf):
    """
    Estimate integrated autocorrelation time.
    τ_int = 0.5 + Σ_{t>0} ACF(t)
    Truncate when ACF drops below threshold to avoid noise.
    """
    tau = 0.5  # Contribution from t=0
    for i in range(1, len(acf)):
        if acf[i] < 0.05:  # Stop when correlation is weak
            break
        tau += acf[i]
    return tau


def binning_analysis(x, n_bins_list):
    """
    Error estimate vs bin size.
    If errors plateau, we've captured the full correlation.
    If errors keep rising, need more data or larger bins.
    """
    errors = []
    for n_bins in n_bins_list:
        bin_size = len(x) // n_bins
        if bin_size < 1:
            continue
        bin_means = [np.mean(x[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)]
        errors.append(np.std(bin_means) / np.sqrt(n_bins))
    return errors


def main():
    # Load data
    try:
        data = np.load(FILE_NAME)
        print(f"Loaded {FILE_NAME}")
    except FileNotFoundError:
        print(f"Error: {FILE_NAME} not found. Run simulation first.")
        return

    Tc = float(data['Tc_final'])
    prod_L = list(data['prod_L'])

    print(f"Tc = {Tc:.5f}")
    print(f"Production lattices: {prod_L}")

    # Check what data is available
    has_M4 = f'prod_L{prod_L[0]}_M4' in data.files
    has_E = f'prod_L{prod_L[0]}_E' in data.files

    if not has_M4:
        print("\nNote: M4 and E time series not found in data.")
        print("Re-run simulation with updated code to get full diagnostics.")
        print("Proceeding with M2 only...\n")

    # Create figure
    n_L = len(prod_L)
    fig, axes = plt.subplots(n_L, 3, figsize=(15, 4*n_L))
    if n_L == 1:
        axes = axes.reshape(1, -1)

    print("\n" + "="*60)
    print("DIAGNOSTICS SUMMARY")
    print("="*60)

    for i, L in enumerate(prod_L):
        M2 = data[f'prod_L{L}_M2']
        V = L * L

        # Normalize M2 for plotting
        M2_norm = M2 / V**2

        # 1. Time series (first 10000 points for visibility)
        ax = axes[i, 0]
        n_show = min(10000, len(M2))
        ax.plot(M2_norm[:n_show], alpha=0.7, lw=0.5)
        ax.set_xlabel('Measurement')
        ax.set_ylabel('M²/V²')
        ax.set_title(f'L={L}: Time Series (first {n_show})')
        ax.grid(alpha=0.3)

        # 2. Autocorrelation
        ax = axes[i, 1]
        acf = autocorrelation(M2, max_lag=500)
        ax.plot(acf)
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.axhline(0.05, color='r', ls=':', lw=0.5, label='threshold')
        tau = integrated_autocorr_time(acf)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title(f'L={L}: Autocorrelation (τ_int ≈ {tau:.1f})')
        ax.set_xlim(0, 200)
        ax.grid(alpha=0.3)

        # 3. Binning analysis
        ax = axes[i, 2]
        n_bins_list = [10, 20, 50, 100, 200, 500, 1000, 2000]
        n_bins_list = [n for n in n_bins_list if n < len(M2) // 10]
        if len(n_bins_list) > 0:
            errors = binning_analysis(M2, n_bins_list)
            bin_sizes = [len(M2) // n for n in n_bins_list[:len(errors)]]
            ax.semilogx(bin_sizes, errors, 'o-')
        ax.set_xlabel('Bin size')
        ax.set_ylabel('Error estimate')
        ax.set_title(f'L={L}: Binning Analysis')
        ax.grid(alpha=0.3)

        # Print summary
        n_eff = len(M2) / (2*tau) if tau > 0 else len(M2)
        print(f"\nL={L}:")
        print(f"  N_samples = {len(M2)}")
        print(f"  <M²>/V² = {np.mean(M2_norm):.6f}")
        print(f"  τ_int ≈ {tau:.1f}")
        print(f"  Effective samples ≈ {n_eff:.0f}")

        # Check for thermalization issues
        # Compare first 20% to last 80%
        n_check = len(M2) // 5
        mean_early = np.mean(M2[:n_check])
        mean_late = np.mean(M2[n_check:])
        drift = abs(mean_early - mean_late) / np.std(M2)
        if drift > 0.5:
            print(f"  WARNING: Possible thermalization issue (drift = {drift:.2f}σ)")
        else:
            print(f"  Thermalization: OK (drift = {drift:.2f}σ)")

    plt.tight_layout()
    plt.savefig('ising2d_diagnostics.png', dpi=150)
    plt.show()

    print("\n" + "="*60)
    print("Plot saved to ising2d_diagnostics.png")
    print("="*60)
    print("\nInterpretation guide:")
    print("- Time series: Should look stationary (no drift)")
    print("- ACF: Should decay to zero; τ_int tells correlation time")
    print("- Binning: Error should plateau; if rising, need more data")
    print("="*60)


if __name__ == "__main__":
    main()
