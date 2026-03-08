import numpy as np
import time
import os
from multiprocessing import Pool, cpu_count

# =============================================================================
# 3D SU(2) LATTICE YANG-MILLS SIMULATION (PARALLEL VERSION)
# =============================================================================
#
# PHYSICAL SETUP:
#   Pure SU(2) gauge theory in 3 Euclidean dimensions. No matter fields.
#   The dynamical variables are SU(2) matrices U_mu(x) living on links.
#   Each link connects site x to site x+mu (mu = 0,1,2 for x,y,z directions).
#
# LATTICE ACTION (Wilson plaquette action):
#   S = -beta * sum_{x,mu<nu} (1/2) Re Tr[P_{mu,nu}(x)]
#
# where P_{mu,nu}(x) is the plaquette (smallest Wilson loop):
#   P_{mu,nu}(x) = U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
#
# This is a 1x1 loop around an elementary square in the mu-nu plane.
# The action penalizes deviation from the identity (flat connection).
#
# CONTINUUM LIMIT:
#   As a -> 0 (lattice spacing), this reproduces the Yang-Mills action:
#   S_continuum = (1/2g^2) * integral d^3x Tr[F_{mu,nu} F^{mu,nu}]
#
# COUPLING:
#   beta = 4/g^2 for SU(2). Larger beta = weaker coupling = closer to continuum.
#   At beta ~ 6, we're in the scaling regime where physics is approximately
#   independent of the UV cutoff (lattice spacing).
#
# MONTE CARLO SAMPLING:
#   We sample gauge configurations {U} from the Boltzmann distribution:
#     P[U] = (1/Z) * exp(-S[U])
#   using the Metropolis algorithm. Observables are computed as ensemble averages.
#
# PARALLELIZATION:
#   Multiple independent Markov chains run in parallel, each collecting
#   a fraction of the total measurements. Results are combined at the end.
#   This gives linear speedup with the number of CPU cores.
#   Independent chains are statistically valid since each explores the
#   same equilibrium distribution after thermalization.
#
# =============================================================================

# --- CONFIGURATION ---
L = 32              # Lattice size (L^3 lattice)
beta = 6.0          # Inverse coupling: beta = 4/g^2 for SU(2)
n_therm = 1000      # Thermalization sweeps before measurement
n_meas = 3000       # Total number of measurement configurations
n_skip = 50         # Sweeps between measurements (decorrelation, should be ~ tau_int)
epsilon_met = 0.3   # Metropolis step size (smaller = higher acceptance)
n_chains = None     # Number of parallel chains (None = auto-detect CPU count)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_cold_start(shape):
    """
    Initialize all links to identity (ordered/cold start).

    Cold start: U_mu(x) = I for all links. This is the classical vacuum
    (zero field strength F=0 everywhere). The system then "heats up"
    during thermalization to reach the quantum equilibrium distribution.

    Alternative: Hot start with random SU(2) matrices. Both converge to
    the same equilibrium, but cold start is often faster for large beta.
    """
    Id = np.zeros(shape + (2, 2), dtype=np.complex128)
    Id[..., 0, 0] = 1.0; Id[..., 1, 1] = 1.0
    return Id

def project_SU2(m):
    """
    Project matrix onto SU(2) using quaternion normalization.

    SU(2) QUATERNION REPRESENTATION:
    Any SU(2) matrix can be written as:
      U = a0*I + i*(a1*sigma_1 + a2*sigma_2 + a3*sigma_3)
    where sigma_i are Pauli matrices and a0^2 + a1^2 + a2^2 + a3^2 = 1.

    In matrix form:
      U = [ a0 + i*a3    a2 + i*a1 ]
          [-a2 + i*a1    a0 - i*a3 ]

    This function extracts (a0,a1,a2,a3) from an arbitrary 2x2 matrix,
    normalizes to the unit 3-sphere, and reconstructs a valid SU(2) matrix.
    Used after smearing to maintain gauge group structure.
    """
    a0 = 0.5 * np.real(m[..., 0, 0] + m[..., 1, 1])
    a1 = 0.5 * np.imag(m[..., 0, 1] + m[..., 1, 0])
    a2 = 0.5 * np.real(m[..., 0, 1] - m[..., 1, 0])
    a3 = 0.5 * np.imag(m[..., 0, 0] - m[..., 1, 1])
    norm = np.sqrt(a0**2 + a1**2 + a2**2 + a3**2)[..., None, None]
    norm = np.where(norm < 1e-10, 1.0, norm)
    a0, a1, a2, a3 = a0/norm[...,0,0], a1/norm[...,0,0], a2/norm[...,0,0], a3/norm[...,0,0]
    U = np.zeros_like(m)
    U[..., 0, 0] = a0 + 1j*a3
    U[..., 0, 1] = a2 + 1j*a1
    U[..., 1, 0] = -a2 + 1j*a1
    U[..., 1, 1] = a0 - 1j*a3
    return U

def random_SU2_updates(shape, epsilon):
    """
    Generate random SU(2) matrices close to identity for Metropolis updates.

    METROPOLIS PROPOSAL DISTRIBUTION:
    We want R close to identity so that U' = R*U is close to U.
    This gives reasonable acceptance rates in the Metropolis algorithm.

    METHOD:
    Generate random quaternion (a0, a1, a2, a3) with:
      - a0 ~ sqrt(1 - epsilon^2)  (close to 1)
      - a1, a2, a3 ~ epsilon * uniform(-0.5, 0.5)
    Then normalize to the unit 3-sphere.

    TUNING:
    - epsilon too small: high acceptance but slow exploration (random walk)
    - epsilon too large: low acceptance, wasted computation
    - Optimal: ~50-70% acceptance rate. epsilon=0.3 typically works well.
    """
    r = np.random.uniform(-0.5, 0.5, shape + (4,))
    # Keep proposals centered near +I; random sign would place half near -I.
    r[..., 0] = np.sqrt(1 - epsilon**2)
    r[..., 1:] *= epsilon
    norm = np.linalg.norm(r, axis=-1, keepdims=True)
    r = r / norm
    a0, a1, a2, a3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    U = np.zeros(shape + (2, 2), dtype=np.complex128)
    U[..., 0, 0] = a0 + 1j*a3; U[..., 0, 1] = a2 + 1j*a1
    U[..., 1, 0] = -a2 + 1j*a1; U[..., 1, 1] = a0 - 1j*a3
    return U

def compute_staples_3d(U, mu):
    """
    Compute the sum of staples for link U_mu(x).

    WHAT IS A STAPLE?
    A staple is the "environment" of a link in the action. For link U_mu(x),
    the staple is the product of the other three links that complete a plaquette:

           x+nu ----U_mu(x+nu)^dag---- x+mu+nu
             |                            |
        U_nu(x)                      U_nu(x+mu)^dag
             |                            |
             x -------[U_mu(x)]------- x+mu

    The plaquette is P = U_mu(x) * staple, so:
      staple_forward = U_nu(x) * U_mu(x+nu) * U_nu(x+mu)^dag

    There's also a "backward" staple in the -nu direction.

    WHY STAPLES?
    The action contribution from link U_mu(x) is:
      S_link = -(beta/2) * Re Tr[U_mu(x) * Staples^dag]

    When we update U_mu(x) -> U'_mu(x), the change in action is:
      dS = -(beta/2) * Re Tr[(U'_mu - U_mu) * Staples^dag]

    This is the key formula for the Metropolis accept/reject step!
    """
    staple_sum = np.zeros_like(U[..., 0, :, :])
    for nu in range(3):
        if nu == mu: continue
        # Forward staple: U_nu(x) * U_mu(x+nu) * U_nu^dag(x+mu)
        U_nu = U[..., nu, :, :]
        U_mu_s = np.roll(U[..., mu, :, :], -1, axis=nu)        # U_mu(x+nu)
        U_nu_dag_s = np.swapaxes(np.roll(U[..., nu, :, :], -1, axis=mu).conj(), -1, -2)  # U_nu^dag(x+mu)
        staple_sum += U_nu @ U_mu_s @ U_nu_dag_s

        # Backward staple: U_nu^dag(x-nu) * U_mu(x-nu) * U_nu(x-nu+mu)
        U_nu_dag_b = np.swapaxes(np.roll(U[..., nu, :, :], 1, axis=nu).conj(), -1, -2)   # U_nu^dag(x-nu)
        U_mu_b = np.roll(U[..., mu, :, :], 1, axis=nu)         # U_mu(x-nu)
        U_nu_b_s = np.roll(np.roll(U[..., nu, :, :], 1, axis=nu), -1, axis=mu)           # U_nu(x-nu+mu)
        staple_sum += U_nu_dag_b @ U_mu_b @ U_nu_b_s
    return staple_sum

def update_metropolis(U):
    """
    Perform one Metropolis sweep over all links.

    =========================================================================
    THIS IS WHERE THE LATTICE ACTION ENTERS THE SIMULATION!
    =========================================================================

    METROPOLIS ALGORITHM:
    For each link U_mu(x):
      1. Propose: U'_mu = R * U_mu where R is random SU(2) close to identity
      2. Compute action change: dS = S[U'] - S[U]
      3. Accept with probability: min(1, exp(-dS))

    ACTION CHANGE FORMULA:
    The Wilson action only depends on U_mu(x) through plaquettes containing it.
    The change in action when U_mu -> U'_mu is:

      dS = -(beta/2) * Re Tr[(U'_mu - U_mu) * Staples^dag]

    This is derived from:
      S = -(beta/2) * sum Re Tr[P] = -(beta/2) * sum Re Tr[U_mu * Staples^dag]

    DETAILED BALANCE:
    The Metropolis acceptance rule satisfies detailed balance:
      P[U] * T(U->U') * A(U->U') = P[U'] * T(U'->U) * A(U'->U)

    Since T(U->U') = T(U'->U) (symmetric proposal), this reduces to:
      A(U->U')/A(U'->U) = P[U']/P[U] = exp(-dS)

    which is satisfied by A = min(1, exp(-dS)).

    ERGODICITY:
    We sweep through all links in all directions to ensure ergodicity
    (any configuration can be reached from any other in finite steps).
    """
    # Build checkerboard sublattices from the current L at call time.
    parity_mask = (
        np.arange(L)[:, None, None]
        + np.arange(L)[None, :, None]
        + np.arange(L)[None, None, :]
    ) & 1
    parity_sublattices = (parity_mask == 0, parity_mask == 1)

    # Checkerboard updates avoid simultaneous updates of neighboring links
    # whose local actions are coupled via staples.
    for mu in range(3):
        for parity_mask in parity_sublattices:
            # Recompute staples after each sublattice update.
            Staples = compute_staples_3d(U, mu)
            Staples_dag = np.swapaxes(Staples.conj(), -1, -2)

            old_link = U[..., mu, :, :]
            R = random_SU2_updates((L, L, L), epsilon=epsilon_met)
            new_link = R @ old_link

            # ACTION CHANGE: dS = -(beta/2) * Re Tr[(U' - U) * Staples^dag]
            dS = -(beta / 2.0) * np.real(
                np.trace((new_link - old_link) @ Staples_dag, axis1=-1, axis2=-2)
            )

            # Metropolis accept/reject on this checkerboard sublattice only.
            accept_prob = np.minimum(1.0, np.exp(-dS))
            r = np.random.uniform(0, 1, dS.shape)
            accept = (r < accept_prob) & parity_mask
            old_link[accept] = new_link[accept]
    return U

# =============================================================================
# SMEARING AND MEASUREMENT FUNCTIONS
# =============================================================================

def spatial_ape_smear(U_in, alpha=0.5, n_steps=1):
    """
    APE smearing in x,y directions only (NOT z).

    WHAT IS SMEARING?
    Smearing replaces each link with a weighted average of itself and
    nearby "staple" paths. This suppresses UV fluctuations (short-wavelength
    noise) while preserving long-distance physics.

    APE SMEARING FORMULA:
      U'_mu = Project_SU2[ (1-alpha)*U_mu + (alpha/n_staples)*sum(staples) ]

    WHY 2D SMEARING ONLY?
    We use the z-direction as Euclidean "time" for the glueball correlator:
      C(t) = <O(0) O(t)>

    If we smear in z, we mix operators at different time slices, corrupting
    the time dependence we're trying to measure. Smearing only in x,y
    improves the signal (better overlap with physical states) while
    preserving the correlator structure.

    MULTIPLE SMEARING LEVELS:
    Different amounts of smearing create different operators that overlap
    with physical states differently. Using several operators in the GEVP
    helps isolate the ground state from excited states.
    """
    U_sm = U_in.copy()
    for _ in range(n_steps):
        U_curr = U_sm.copy()
        for mu in range(2):  # Only x,y directions (NOT z!)
            staple_sum = np.zeros_like(U_curr[..., mu, :, :])
            for nu in range(2):  # Only x,y directions
                if mu == nu: continue
                U_nu = U_curr[..., nu, :, :]
                U_mu_s = np.roll(U_curr[..., mu, :, :], -1, axis=nu)
                U_nu_dag_s = np.swapaxes(np.roll(U_curr[..., nu, :, :], -1, axis=mu).conj(), -1, -2)
                staple_sum += U_nu @ U_mu_s @ U_nu_dag_s
                U_nu_dag_b = np.swapaxes(np.roll(U_curr[..., nu, :, :], 1, axis=nu).conj(), -1, -2)
                U_mu_b = np.roll(U_curr[..., mu, :, :], 1, axis=nu)
                U_nu_b_s = np.roll(np.roll(U_curr[..., nu, :, :], 1, axis=nu), -1, axis=mu)
                staple_sum += U_nu_dag_b @ U_mu_b @ U_nu_b_s
            # Weighted average: (1-alpha)*link + (alpha/2)*staples
            U_temp = (1.0 - alpha) * U_curr[..., mu, :, :] + (alpha / 2.0) * staple_sum
            # Project back to SU(2) to maintain gauge group structure
            U_sm[..., mu, :, :] = project_SU2(U_temp)
    return U_sm

def measure_glueball_3d(U):
    """
    Measure glueball operator: sum of plaquette traces in the x-y plane.

    GLUEBALL OPERATOR:
    The 0++ glueball (scalar, positive parity, positive charge conjugation)
    couples to the operator:
      O(z) = sum_{x,y} Re Tr[P_{01}(x,y,z)]

    This is the sum of all plaquettes in the x-y plane at fixed z.
    The trace is real for SU(2) and gauge-invariant.

    CORRELATOR:
    We compute C(t) = <O(0) O(t)> which decays as:
      C(t) ~ A * exp(-m_G * t) for large t
    where m_G is the glueball mass.

    The glueball is a "gluonic bound state" - a massive excitation of the
    pure gauge field with no quarks. In 3D SU(2), the lightest glueball
    is the 0++ with m_G/sqrt(sigma) ~ 4-5.
    """
    U_0 = U[..., 0, :, :]  # Links in x-direction
    U_1 = U[..., 1, :, :]  # Links in y-direction
    # Plaquette P_{01} = U_0(x) * U_1(x+0) * U_0^dag(x+1) * U_1^dag(x)
    P01 = (U_0 @ np.roll(U_1, -1, axis=0)) @ (np.swapaxes(np.roll(U_0, -1, axis=1).conj(),-1,-2) @ np.swapaxes(U_1.conj(),-1,-2))
    trP = np.real(np.trace(P01, axis1=-2, axis2=-1))
    # Sum over x,y at each z to get O(z)
    return np.sum(trP, axis=(0, 1))

def measure_wilson_loops_3d(U, R_max, T_max):
    """
    Measure Wilson loops W(R,T) for extracting the static quark potential.

    WILSON LOOP:
    W(R,T) = <(1/2) Re Tr[Loop]> where Loop is an R x T rectangle:

        x ----R---- x+R
        |          |
        T          T
        |          |
        x ----R---- x+R

    PHYSICAL INTERPRETATION:
    W(R,T) = <exp(-i * g * integral A.dl)> around the loop.

    For large T, this probes the energy of a static quark-antiquark pair
    separated by distance R:
      W(R,T) ~ exp(-V(R) * T)  as T -> infinity

    The static potential V(R) can be extracted from:
      V(R) = -log(W(R,T+1)/W(R,T)) = log(W(R,T)/W(R,T+1))

    CONFINEMENT:
    In a confining theory, V(R) grows linearly at large R:
      V(R) = sigma * R + const + O(1/R)

    The string tension sigma is the energy per unit length of the
    chromoelectric flux tube connecting the quarks. This is the
    definitive signature of confinement.

    NOTE: We use R in x-direction (mu=0) and T in z-direction (nu=2).
    Spatial links are smeared, temporal links are not (hybrid approach).
    """
    W_RT = np.zeros((R_max, T_max))
    mu, nu = 0, 2  # Spatial (R) and temporal (T) directions
    U_R_line = U[..., mu, :, :]
    for r in range(1, R_max + 1):
        # Build spatial line of length r: U_0(x) * U_0(x+1) * ... * U_0(x+r-1)
        spatial_line = U_R_line.copy()
        U_shift = np.roll(U[..., mu, :, :], -r, axis=mu)
        U_R_line = U_R_line @ U_shift
        U_T_line = U[..., nu, :, :]
        for t in range(1, T_max + 1):
            # Build Wilson loop: bottom * right * top^dag * left^dag
            V_0 = U_T_line.copy()           # Left side (temporal links at x)
            V_R = np.roll(V_0, -r, axis=mu) # Right side (temporal links at x+R)
            top = np.roll(spatial_line, -t, axis=nu)  # Top (spatial links at t)
            top_dag = np.swapaxes(top.conj(), -1, -2)
            V_0_dag = np.swapaxes(V_0.conj(), -1, -2)
            # Complete the loop
            Loop = spatial_line @ V_R @ top_dag @ V_0_dag
            val = np.real(np.mean(np.trace(Loop, axis1=-2, axis2=-1)))
            W_RT[r-1, t-1] += val
            # Extend temporal line for next iteration
            U_time_shift = np.roll(U[..., nu, :, :], -t, axis=nu)
            U_T_line = U_T_line @ U_time_shift
    return W_RT

# =============================================================================
# WORKER FUNCTION FOR PARALLEL CHAINS
# =============================================================================

def run_chain(args):
    """
    Run a single Markov chain. Called by multiprocessing pool.
    Each chain thermalizes independently and collects measurements.

    MARKOV CHAIN MONTE CARLO:
    We generate a sequence of configurations U_1 -> U_2 -> U_3 -> ...
    using the Metropolis algorithm. After thermalization, each U_i is
    a sample from the Boltzmann distribution P[U] ~ exp(-S[U]).

    Observables are computed as time averages:
      <O> = lim (1/N) sum_{i=1}^N O[U_i]

    THERMALIZATION:
    Starting from a cold start (all links = identity), the system must
    "heat up" to reach equilibrium. During this phase, observables drift
    systematically. We discard these configurations.

    DECORRELATION:
    Successive configurations are correlated (they differ by only one
    Metropolis sweep). We skip n_skip sweeps between measurements to
    reduce autocorrelation. Ideally n_skip ~ tau_int (integrated
    autocorrelation time).
    """
    chain_id, n_meas_per_chain, seed = args

    # Set unique random seed for this chain (critical for parallel runs!)
    np.random.seed(seed)

    # =========================================================================
    # THERMALIZATION PHASE
    # =========================================================================
    # Run many Metropolis sweeps without measuring to reach equilibrium.
    # The system evolves from the initial state (cold start) to a typical
    # configuration drawn from exp(-S[U]).
    print(f"  Chain {chain_id}: thermalizing ({n_therm} sweeps)...", flush=True)
    t_start = time.time()
    U = get_cold_start((L, L, L, 3))  # Start from ordered configuration
    for _ in range(n_therm):
        U = update_metropolis(U)       # Each sweep updates all links once
    t_therm = time.time() - t_start
    print(f"  Chain {chain_id}: thermalization done in {t_therm:.1f}s, starting measurements...", flush=True)

    # Storage for this chain
    smear_levels = [10, 20, 30]  # Different smearing levels for GEVP
    n_ops = len(smear_levels)
    ops_history = np.zeros((n_meas_per_chain, n_ops, L))
    R_max, T_max = 6, 6
    wilson_history = np.zeros((n_meas_per_chain, R_max, T_max))

    # =========================================================================
    # MEASUREMENT PHASE
    # =========================================================================
    t_meas_start = time.time()
    for i in range(n_meas_per_chain):
        # Decorrelation: skip n_skip sweeps between measurements
        # This reduces autocorrelation between successive measurements
        for _ in range(n_skip):
            U = update_metropolis(U)

        # -----------------------------------------------------------------
        # GEVP MEASUREMENTS: Glueball operators at different smearing levels
        # -----------------------------------------------------------------
        # Multiple operators with different smearing overlap differently with
        # physical states. The GEVP (Generalized Eigenvalue Problem) uses this
        # to optimally extract the ground state mass.
        U_curr = U.copy()
        U_curr = spatial_ape_smear(U_curr, alpha=0.5, n_steps=smear_levels[0])
        ops_history[i, 0, :] = measure_glueball_3d(U_curr)  # 10 smearing steps

        U_curr = spatial_ape_smear(U_curr, alpha=0.5, n_steps=smear_levels[1]-smear_levels[0])
        ops_history[i, 1, :] = measure_glueball_3d(U_curr)  # 20 smearing steps total

        U_curr = spatial_ape_smear(U_curr, alpha=0.5, n_steps=smear_levels[2]-smear_levels[1])
        ops_history[i, 2, :] = measure_glueball_3d(U_curr)  # 30 smearing steps total

        # -----------------------------------------------------------------
        # WILSON LOOP MEASUREMENTS: For string tension extraction
        # -----------------------------------------------------------------
        # Use "hybrid" smearing: smear spatial links (x,y) but keep
        # temporal links (z) unsmeared. This improves signal for the
        # spatial extent while preserving the area law decay in time.
        U_space_smeared = spatial_ape_smear(U, alpha=0.5, n_steps=10)
        U_hybrid = U.copy()
        U_hybrid[..., 0, :, :] = U_space_smeared[..., 0, :, :]  # Smeared x-links
        U_hybrid[..., 1, :, :] = U_space_smeared[..., 1, :, :]  # Smeared y-links
        # U_hybrid[..., 2, :, :] keeps original z-links (temporal direction)
        wilson_history[i] = measure_wilson_loops_3d(U_hybrid, R_max, T_max)

        # Progress output every 50 measurements
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_meas_start
            rate = (i + 1) / elapsed
            remaining = (n_meas_per_chain - i - 1) / rate if rate > 0 else 0
            print(f"  Chain {chain_id}: {i+1}/{n_meas_per_chain} measurements ({remaining/60:.1f} min left)", flush=True)

    return chain_id, ops_history, wilson_history

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print(f"--- 3D SU(2) Data Collection Run (PARALLEL) ---")

    # =========================================================================
    # PARALLEL CHAIN SETUP
    # =========================================================================
    # We run multiple independent Markov chains in parallel. This is valid
    # because each chain, after thermalization, samples from the same
    # equilibrium distribution. The combined measurements are statistically
    # equivalent to a single long chain (with better error properties since
    # different chains explore different regions of configuration space).

    if n_meas < 1:
        raise ValueError("n_meas must be >= 1")

    if n_chains is None:
        num_chains = max(1, cpu_count() - 1)  # Leave one core free
    else:
        num_chains = n_chains
    num_chains = min(num_chains, n_meas)

    # Distribute measurements across chains, preserving total exactly.
    base_count, remainder = divmod(n_meas, num_chains)
    meas_counts = [base_count + (1 if i < remainder else 0) for i in range(num_chains)]
    total_meas = int(np.sum(meas_counts))

    print(f"Using {num_chains} parallel chains")
    print(f"Measurements/chain (min/max): {min(meas_counts)}/{max(meas_counts)}")
    print(f"Total measurements: {total_meas}")
    print(f"Parameters: L={L}, beta={beta}, n_therm={n_therm}, n_skip={n_skip}")

    # Each chain needs a unique random seed to ensure independence
    base_seed = int(time.time()) % 100000
    chain_args = [(i, meas_counts[i], base_seed + i * 1000) for i in range(num_chains)]

    # =========================================================================
    # RUN PARALLEL SIMULATION
    # =========================================================================
    t0 = time.time()
    print(f"Starting parallel simulation...")

    # multiprocessing.Pool distributes work across CPU cores
    with Pool(num_chains) as pool:
        results = pool.map(run_chain, chain_args)

    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed/60:.1f} minutes")

    # =========================================================================
    # COMBINE RESULTS
    # =========================================================================
    # Concatenate operator measurements from all chains
    # Average Wilson loops (each chain measured the same loops)
    smear_levels = [10, 20, 30]
    n_ops = len(smear_levels)
    ops_history_combined = np.zeros((total_meas, n_ops, L))
    R_max, T_max = 6, 6
    wilson_history_combined = np.zeros((total_meas, R_max, T_max))

    offset = 0
    for chain_id, ops_history, wilson_history in results:
        n_from_chain = ops_history.shape[0]
        ops_history_combined[offset:offset+n_from_chain] = ops_history
        wilson_history_combined[offset:offset+n_from_chain] = wilson_history
        offset += n_from_chain
        print(f"  Chain {chain_id}: collected {n_from_chain} measurements")

    wilson_avg = np.mean(wilson_history_combined, axis=0)

    # =========================================================================
    # SAVE DATA
    # =========================================================================
    # Save raw data for offline analysis. The analysis script will:
    # 1. Build GEVP correlation matrices and extract glueball mass
    # 2. Fit Wilson loops to extract string tension
    # 3. Compute the dimensionless ratio m_G/sqrt(sigma)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'lattice_data_3d.npz')
    print(f"Saving raw data to '{output_path}'...")
    np.savez(output_path,
             ops_history=ops_history_combined,
             wilson_history=wilson_history_combined,
             wilson_avg=wilson_avg,
             beta=beta, L=L)
    print("Done. Use the analysis script to fit.")
