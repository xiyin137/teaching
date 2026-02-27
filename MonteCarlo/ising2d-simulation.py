"""
2D Ising Model Monte Carlo Simulation using Wolff Cluster Algorithm
Computes critical temperature and exponents via finite-size scaling

=============================================================================
THE ISING MODEL
=============================================================================
The 2D Ising model describes spins s_i = ±1 on a square lattice with the
Hamiltonian (action):

    H = -J * Σ_{<i,j>} s_i * s_j

where <i,j> denotes nearest-neighbor pairs and J > 0 (ferromagnetic coupling).
We set J = 1 throughout.

The partition function is Z = Σ_{configs} exp(-βH) where β = 1/(k_B T).

Key observables:
  - Magnetization: M = Σ_i s_i
  - Energy: E = -Σ_{<i,j>} s_i s_j
  - Susceptibility: χ = β<M²>/V  (measures fluctuations)
  - Binder cumulant: U4 = 1 - <M⁴>/(3<M²>²)  (dimensionless, for Tc)

At the critical point Tc = 2/ln(1+√2) ≈ 2.269, the system exhibits:
  - Power-law decay of correlations: G(r) ~ r^(-η) with η = 1/4
  - Diverging susceptibility: χ ~ L^(γ/ν) with γ/ν = 7/4
  - Vanishing magnetization: <|m|> ~ L^(-β/ν) with β/ν = 1/8

=============================================================================
ALGORITHM: WOLFF CLUSTER
=============================================================================
The Wolff algorithm efficiently samples near Tc by flipping entire clusters:

1. Pick a random seed spin
2. Grow cluster: add aligned neighbors with probability p = 1 - exp(-2βJ)
3. Flip all spins in the cluster

This avoids critical slowing down (τ ~ L^z with z ≈ 0.25 vs z ≈ 2 for local)

=============================================================================
SIMULATION PHASES
=============================================================================
1. SCOUT: Coarse Tc estimate from Binder crossing of small lattices
2. REFINE: Precise Tc from larger lattices with finer T grid
3. PRODUCTION: Measure χ, <M²> at Tc for multiple L for scaling
4. CORRELATION: Measure G(r) at Tc for η extraction
5. PHASE SCAN: M vs T for phase transition visualization
"""
import numpy as np
from numba import jit
import multiprocessing
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_FILE = "ising_data.npz"

# Lattice sizes for different phases
SCOUT_L = [48, 96]         # Larger for better Tc estimate
REFINE_L = [96, 128]       # Larger for more accurate crossing
PROD_L = [16, 32, 48, 64, 96, 128]

# Monte Carlo steps
SCOUT_STEPS = 30000
REFINE_STEPS = 50000       # More steps for better statistics
PROD_STEPS = 50000
CORR_STEPS = 20000         # Balance between stats and speed
PHASE_STEPS = 10000        # For M vs T scan

# Thermalization (in sweeps)
THERM_SWEEPS = 2000        # More thermalization

# Phase transition scan
PHASE_L = 64               # Lattice size for M vs T plot


def make_seed():
    """Generate random seed from OS entropy."""
    return int.from_bytes(os.urandom(4), 'little') % (2**31 - 1)


# ============================================================================
# WOLFF CLUSTER ALGORITHM
# ============================================================================
# The Wolff algorithm grows clusters of aligned spins and flips them together.
#
# CONNECTION TO ISING ACTION:
# ---------------------------
# The Ising Hamiltonian is H = -J Σ_{<i,j>} s_i s_j
#
# For two aligned spins (s_i = s_j), flipping one creates an energy cost:
#   ΔE = 2J  (bond goes from -J to +J)
#
# The Boltzmann weight ratio is exp(-β ΔE) = exp(-2βJ)
#
# To satisfy detailed balance, we activate bonds between aligned spins with
# probability:
#   p = 1 - exp(-2βJ)
#
# This ensures: P(old → new) / P(new → old) = exp(-β ΔH)
#
# Key insight: The probability p encodes the Ising interaction strength J
# and temperature T through β = 1/T. At high T (small β), p → 0 (small clusters).
# At low T (large β), p → 1 (large clusters). At Tc, clusters are fractal.
# ============================================================================
@jit(nopython=True, cache=True)
def wolff_cluster(spins, L, prob):
    """
    Perform one Wolff cluster flip.
    prob = 1 - exp(-2*beta*J) is the bond activation probability.
    Returns number of spins flipped.
    """
    # Pick random starting site
    x0 = np.random.randint(0, L)
    y0 = np.random.randint(0, L)
    cluster_spin = spins[x0, y0]

    # Stack for cluster growth (BFS)
    stack = [(x0, y0)]
    spins[x0, y0] *= -1  # Flip immediately to mark as visited
    flipped = 1

    while len(stack) > 0:
        x, y = stack.pop()

        # Check all 4 neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = (x + dx) % L
            ny = (y + dy) % L

            # If neighbor has same spin as original cluster spin and not yet flipped
            if spins[nx, ny] == cluster_spin:
                if np.random.random() < prob:
                    spins[nx, ny] *= -1  # Flip to mark visited
                    stack.append((nx, ny))
                    flipped += 1

    return flipped


@jit(nopython=True, cache=True)
def mc_sweep(spins, L, prob):
    """Perform enough Wolff steps to flip approximately N spins (one sweep)."""
    N = L * L
    total_flipped = 0
    while total_flipped < N:
        total_flipped += wolff_cluster(spins, L, prob)
    return total_flipped


@jit(nopython=True, cache=True)
def compute_observables(spins, L):
    """Compute magnetization and energy."""
    M = 0.0
    E = 0.0
    for x in range(L):
        for y in range(L):
            s = spins[x, y]
            M += s
            # Only count bonds in +x and +y directions to avoid double counting
            E -= s * spins[(x + 1) % L, y]
            E -= s * spins[x, (y + 1) % L]
    return M, E


@jit(nopython=True, cache=True)
def run_simulation(L, beta, n_steps, n_therm, seed):
    """
    Run Monte Carlo simulation at given temperature.
    Returns arrays of M^2, M^4, and E for each measurement.
    """
    np.random.seed(seed)

    # Initialize spins (all +1)
    spins = np.ones((L, L), dtype=np.int8)

    # Bond probability for Wolff algorithm: p = 1 - exp(-2βJ)
    # This is where the Ising action H = -J Σ s_i s_j enters the simulation.
    # J = 1 is implicit; β = 1/T controls the coupling strength.
    prob = 1.0 - np.exp(-2.0 * beta)

    # Thermalization
    for _ in range(n_therm):
        mc_sweep(spins, L, prob)

    # Measurement
    M2_arr = np.zeros(n_steps, dtype=np.float64)
    M4_arr = np.zeros(n_steps, dtype=np.float64)
    E_arr = np.zeros(n_steps, dtype=np.float64)

    for i in range(n_steps):
        mc_sweep(spins, L, prob)
        M, E = compute_observables(spins, L)
        M2_arr[i] = M * M
        M4_arr[i] = M * M * M * M
        E_arr[i] = E

    return M2_arr, M4_arr, E_arr


# ============================================================================
# BINDER CUMULANT AND TC FINDING
# ============================================================================
def binder_cumulant(M2_arr, M4_arr):
    """Compute Binder cumulant U4 = 1 - <M^4>/(3<M^2>^2)."""
    m2_avg = np.mean(M2_arr)
    m4_avg = np.mean(M4_arr)
    if m2_avg == 0:
        return 0.0
    return 1.0 - m4_avg / (3.0 * m2_avg * m2_avg)


def find_crossing(T_arr, U4_L1, U4_L2):
    """Find temperature where Binder cumulants cross using linear interpolation."""
    diff = U4_L1 - U4_L2
    for i in range(len(T_arr) - 1):
        if diff[i] * diff[i + 1] < 0:  # Sign change
            # Linear interpolation
            T1, T2 = T_arr[i], T_arr[i + 1]
            d1, d2 = diff[i], diff[i + 1]
            Tc = T1 - d1 * (T2 - T1) / (d2 - d1)
            return Tc
    return None


def worker_simulate(args):
    """Worker function for parallel simulation."""
    L, T, n_steps, n_therm, seed, verbose = args if len(args) == 6 else (*args, False)
    if verbose:
        print(f"  Starting L={L}, T={T:.4f}...", flush=True)
    beta = 1.0 / T
    M2, M4, E = run_simulation(L, beta, n_steps, n_therm, seed)
    if verbose:
        print(f"  Done L={L}, T={T:.4f}", flush=True)
    return L, T, M2, M4, E


# ============================================================================
# CORRELATION FUNCTION (FFT-based)
# ============================================================================
def compute_correlation_fft(spins):
    """Compute G(r) using FFT: G = IFFT(|FFT(s)|^2) / N."""
    L = spins.shape[0]
    N = L * L
    s = spins.astype(np.float64)
    fft_s = np.fft.fft2(s)
    corr = np.fft.ifft2(np.abs(fft_s) ** 2).real / N
    return corr


def run_correlation_measurement(L, T, n_steps, n_therm, seed):
    """Run correlation measurement with FFT."""
    np.random.seed(seed)  # For initial state

    spins = np.ones((L, L), dtype=np.int8)
    beta = 1.0 / T
    prob = 1.0 - np.exp(-2.0 * beta)  # Ising action enters here via β = 1/T
    max_r = L // 2

    # Precompute distance bins
    g_count = np.zeros(max_r + 1, dtype=np.float64)
    for dx in range(L):
        ddx = min(dx, L - dx)
        for dy in range(L):
            ddy = min(dy, L - dy)
            r = int(np.sqrt(ddx * ddx + ddy * ddy) + 0.5)
            if r <= max_r:
                g_count[r] += 1.0

    # JIT-compiled thermalization
    @jit(nopython=True, cache=True)
    def thermalize_and_sweep(spins, L, prob, n_therm, seed):
        np.random.seed(seed)
        for _ in range(n_therm):
            mc_sweep(spins, L, prob)

    @jit(nopython=True, cache=True)
    def do_sweep(spins, L, prob):
        mc_sweep(spins, L, prob)

    thermalize_and_sweep(spins, L, prob, n_therm, seed)

    # Accumulate correlation
    g_sum = np.zeros(max_r + 1, dtype=np.float64)
    report_interval = max(1, n_steps // 10)

    for step in range(n_steps):
        if step % report_interval == 0:
            print(f"  Correlation: {100*step//n_steps}% done...", flush=True)
        do_sweep(spins, L, prob)
        corr = compute_correlation_fft(spins)

        # Bin by distance
        for dx in range(L):
            ddx = min(dx, L - dx)
            for dy in range(L):
                ddy = min(dy, L - dy)
                r = int(np.sqrt(ddx * ddx + ddy * ddy) + 0.5)
                if r <= max_r:
                    g_sum[r] += corr[dx, dy]

    # Average over steps
    g_avg = g_sum / (n_steps * g_count)
    g_avg[g_count == 0] = 0

    return g_avg, g_count


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("2D ISING MODEL - MONTE CARLO SIMULATION")
    print("=" * 60)

    # Warmup JIT compilation
    print("\nCompiling JIT kernels...")
    dummy = np.ones((4, 4), dtype=np.int8)
    wolff_cluster(dummy, 4, 0.5)
    run_simulation(4, 0.5, 10, 10, 12345)

    # ========================================================================
    # PHASE 1: SCOUT - Find approximate Tc using Binder crossing
    # ========================================================================
    print(f"\n--- Phase 1: Scout (L={SCOUT_L}) ---")

    # Wide temperature scan to find crossing (no prior knowledge of Tc)
    T_scan = np.linspace(2.0, 2.5, 21)

    # Run simulations
    tasks = []
    for T in T_scan:
        for L in SCOUT_L:
            tasks.append((L, T, SCOUT_STEPS, THERM_SWEEPS, make_seed()))

    print(f"Running {len(tasks)} simulations...")
    with multiprocessing.Pool() as pool:
        results = pool.map(worker_simulate, tasks)

    # Organize results by L and T
    data = {L: {} for L in SCOUT_L}
    for L, T, M2, M4, E in results:
        data[L][T] = {'M2': M2, 'M4': M4, 'E': E, 'U4': binder_cumulant(M2, M4)}

    # Print Binder cumulants
    print(f"\n  T       U4(L={SCOUT_L[0]})  U4(L={SCOUT_L[1]})  diff")
    print("  " + "-" * 45)
    U4_L1 = np.array([data[SCOUT_L[0]][T]['U4'] for T in T_scan])
    U4_L2 = np.array([data[SCOUT_L[1]][T]['U4'] for T in T_scan])
    for i, T in enumerate(T_scan):
        print(f"  {T:.3f}   {U4_L1[i]:.4f}        {U4_L2[i]:.4f}        {U4_L1[i]-U4_L2[i]:+.5f}")

    # Find crossing
    Tc_scout = find_crossing(T_scan, U4_L1, U4_L2)
    if Tc_scout is None:
        print("WARNING: No crossing found! Using 2.269 as fallback.")
        Tc_scout = 2.269
    print(f"\nScout Tc = {Tc_scout:.6f}")

    # ========================================================================
    # PHASE 2: REFINE - Higher precision Tc using larger lattices
    # ========================================================================
    print(f"\n--- Phase 2: Refine (L={REFINE_L}) ---")

    # Narrower scan around scout estimate
    T_refine = np.linspace(Tc_scout - 0.02, Tc_scout + 0.02, 11)

    tasks = []
    for T in T_refine:
        for L in REFINE_L:
            tasks.append((L, T, REFINE_STEPS, THERM_SWEEPS, make_seed(), True))

    print(f"Running {len(tasks)} simulations (this may take a while for L=128)...")
    with multiprocessing.Pool() as pool:
        results = pool.map(worker_simulate, tasks)

    # Organize results
    data_ref = {L: {} for L in REFINE_L}
    for L, T, M2, M4, E in results:
        data_ref[L][T] = {'M2': M2, 'M4': M4, 'E': E, 'U4': binder_cumulant(M2, M4)}

    U4_ref1 = np.array([data_ref[REFINE_L[0]][T]['U4'] for T in T_refine])
    U4_ref2 = np.array([data_ref[REFINE_L[1]][T]['U4'] for T in T_refine])

    Tc_final = find_crossing(T_refine, U4_ref1, U4_ref2)
    if Tc_final is None:
        print("WARNING: No crossing found in refine! Using scout Tc.")
        Tc_final = Tc_scout
    print(f"Final Tc = {Tc_final:.6f}")

    # ========================================================================
    # PHASE 3: PRODUCTION - Susceptibility scaling at Tc
    # ========================================================================
    print(f"\n--- Phase 3: Production at Tc = {Tc_final:.6f} ---")

    tasks = []
    for L in PROD_L:
        tasks.append((L, Tc_final, PROD_STEPS, THERM_SWEEPS, make_seed(), True))

    print(f"Running {len(tasks)} simulations...")
    with multiprocessing.Pool() as pool:
        results_prod = pool.map(worker_simulate, tasks)

    # Store production data (including E and M4 for diagnostics)
    prod_data = {}
    for L, T, M2, M4, E in results_prod:
        prod_data[L] = {'M2': M2, 'M4': M4, 'E': E}
        chi = np.mean(M2) / (L * L * Tc_final)  # χ = β<M²>/V
        print(f"  L={L:3d}: χ = {chi:.2f}")

    # ========================================================================
    # PHASE 4: CORRELATION FUNCTION at Tc
    # ========================================================================
    L_corr = 128  # Good balance of range vs speed
    print(f"\n--- Phase 4: Correlation Function (L={L_corr}) ---")
    print(f"Running correlation measurement...")
    g_r, g_count = run_correlation_measurement(L_corr, Tc_final, CORR_STEPS, THERM_SWEEPS, make_seed())

    # ========================================================================
    # PHASE 5: PHASE TRANSITION SCAN (M vs T)
    # ========================================================================
    print(f"\n--- Phase 5: Phase Transition Scan (L={PHASE_L}) ---")

    T_phase = np.linspace(1.5, 3.5, 41)  # Wide range around Tc
    M_phase = []
    M_phase_err = []

    for T in T_phase:
        print(f"  T = {T:.3f}...", end=" ", flush=True)
        beta = 1.0 / T
        M2, M4, E = run_simulation(PHASE_L, beta, PHASE_STEPS, THERM_SWEEPS, make_seed())
        # <|M|> = sqrt(<M²>) / V  (since <M>=0 by symmetry)
        V = PHASE_L * PHASE_L
        m_samples = np.sqrt(M2) / V
        m_avg = np.mean(m_samples)
        m_err = np.std(m_samples) / np.sqrt(len(m_samples))
        M_phase.append(m_avg)
        M_phase_err.append(m_err)
        print(f"<|m|> = {m_avg:.4f}", flush=True)

    M_phase = np.array(M_phase)
    M_phase_err = np.array(M_phase_err)

    # ========================================================================
    # SAVE DATA
    # ========================================================================
    print(f"\nSaving to {OUTPUT_FILE}...")

    save_dict = {
        'Tc_scout': Tc_scout,
        'Tc_final': Tc_final,
        'refine_L': REFINE_L,
        'refine_T': T_refine,
        'prod_L': PROD_L,
        'corr_L': L_corr,
        'corr_g': g_r,
        'corr_count': g_count,
        # Phase transition data
        'phase_L': PHASE_L,
        'phase_T': T_phase,
        'phase_M': M_phase,
        'phase_M_err': M_phase_err,
    }

    # Add refine data for Binder plot
    for L in REFINE_L:
        for T in T_refine:
            key = f'ref_L{L}_T{T:.4f}'
            save_dict[f'{key}_M2'] = data_ref[L][T]['M2']
            save_dict[f'{key}_M4'] = data_ref[L][T]['M4']
            save_dict[f'{key}_E'] = data_ref[L][T]['E']

    # Add production data (full time series for diagnostics)
    for L in PROD_L:
        save_dict[f'prod_L{L}_M2'] = prod_data[L]['M2']
        save_dict[f'prod_L{L}_M4'] = prod_data[L]['M4']
        save_dict[f'prod_L{L}_E'] = prod_data[L]['E']

    np.savez_compressed(OUTPUT_FILE, **save_dict)
    print("Done!")
