#!/usr/bin/env python3
"""
Run one Monte Carlo shard (one chain id) for latticeYM3d simulation.

This wrapper loads MonteCarlo/latticeYM3d-simulation.py, overrides selected
globals, executes run_chain(), and writes a shard npz file.
"""
import argparse
import importlib.util
import os
import time

import numpy as np


def load_simulation_module(simulation_script):
    spec = importlib.util.spec_from_file_location("latticeYM3d_simulation", simulation_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load simulation module from {simulation_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measurements_for_chain(total_meas, total_chains, chain_id):
    base, rem = divmod(total_meas, total_chains)
    return base + (1 if chain_id < rem else 0)


def main():
    parser = argparse.ArgumentParser(description="Run one latticeYM3d chain shard.")
    default_sim = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "latticeYM3d-simulation.py",
    )
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "shards",
    )
    parser.add_argument("--simulation-script", default=default_sim)
    parser.add_argument("--chain-id", type=int, required=True)
    parser.add_argument("--total-chains", type=int, required=True)
    parser.add_argument("--total-meas", type=int, default=3000)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--out-dir", default=default_out)

    # Optional overrides for simulation globals.
    parser.add_argument("--lattice-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--n-therm", type=int, default=None)
    parser.add_argument("--n-skip", type=int, default=None)
    parser.add_argument("--epsilon-met", type=float, default=None)
    args = parser.parse_args()

    if args.total_chains < 1:
        raise ValueError("--total-chains must be >= 1")
    if args.total_meas < 1:
        raise ValueError("--total-meas must be >= 1")
    if args.chain_id < 0 or args.chain_id >= args.total_chains:
        raise ValueError("--chain-id must satisfy 0 <= chain-id < total-chains")

    n_meas_chain = measurements_for_chain(args.total_meas, args.total_chains, args.chain_id)
    if n_meas_chain < 1:
        raise ValueError(
            f"Chain {args.chain_id} received n_meas_chain={n_meas_chain}; "
            "increase total_meas or reduce total_chains."
        )

    sim = load_simulation_module(args.simulation_script)

    if args.lattice_size is not None:
        sim.L = args.lattice_size
    if args.beta is not None:
        sim.beta = args.beta
    if args.n_therm is not None:
        sim.n_therm = args.n_therm
    if args.n_skip is not None:
        sim.n_skip = args.n_skip
    if args.epsilon_met is not None:
        sim.epsilon_met = args.epsilon_met

    seed = args.seed_base + args.chain_id * 1000

    print(
        "Running shard:",
        f"chain_id={args.chain_id}",
        f"n_meas_chain={n_meas_chain}",
        f"L={sim.L}",
        f"beta={sim.beta}",
        f"n_therm={sim.n_therm}",
        f"n_skip={sim.n_skip}",
        f"epsilon_met={sim.epsilon_met}",
        f"seed={seed}",
    )
    t0 = time.time()
    chain_id_out, ops_history, wilson_history = sim.run_chain((args.chain_id, n_meas_chain, seed))
    elapsed = time.time() - t0
    if chain_id_out != args.chain_id:
        raise RuntimeError(f"Chain id mismatch: expected {args.chain_id}, got {chain_id_out}")

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, f"shard_{args.chain_id:04d}.npz")
    np.savez_compressed(
        output_path,
        chain_id=args.chain_id,
        total_chains=args.total_chains,
        total_meas_requested=args.total_meas,
        n_meas_chain=n_meas_chain,
        seed=seed,
        L=int(sim.L),
        beta=float(sim.beta),
        n_therm=int(sim.n_therm),
        n_skip=int(sim.n_skip),
        epsilon_met=float(sim.epsilon_met),
        ops_history=ops_history,
        wilson_history=wilson_history,
    )
    print(
        f"Shard complete in {elapsed:.1f}s. "
        f"Saved {ops_history.shape[0]} configs to {output_path}"
    )


if __name__ == "__main__":
    main()
