#!/usr/bin/env python3
"""
Merge latticeYM3d shard outputs into a single lattice_data_3d.npz file.

Expected shard format is produced by run_latticeYM3d_shard.py.
"""
import argparse
import glob
import os
import re

import numpy as np


def parse_chain_id_from_name(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"Could not parse chain id from filename: {name}")
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser(description="Merge latticeYM3d shard npz files.")
    default_input = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "shards",
    )
    default_output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "lattice_data_3d_merged.npz",
    )
    parser.add_argument("--input-dir", default=default_input)
    parser.add_argument("--pattern", default="shard_*.npz")
    parser.add_argument("--expected-chains", type=int, default=None)
    parser.add_argument("--output", default=default_output)
    parser.add_argument(
        "--delete-input-shards",
        action="store_true",
        help="Delete shard files after successful merge.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise FileNotFoundError(
            f"No shard files found in {args.input_dir!r} with pattern {args.pattern!r}"
        )

    entries = []
    for path in files:
        with np.load(path) as data:
            if "chain_id" in data.files:
                chain_id = int(data["chain_id"])
            else:
                chain_id = parse_chain_id_from_name(path)
            entry = {
                "path": path,
                "chain_id": chain_id,
                "L": int(data["L"]),
                "beta": float(data["beta"]),
                "ops_history": data["ops_history"],
                "wilson_history": data["wilson_history"],
                "n_meas_chain": int(data["n_meas_chain"]) if "n_meas_chain" in data.files else None,
            }
            entries.append(entry)

    entries.sort(key=lambda x: x["chain_id"])
    chain_ids = [e["chain_id"] for e in entries]
    if len(chain_ids) != len(set(chain_ids)):
        raise ValueError(f"Duplicate chain ids found: {chain_ids}")

    if args.expected_chains is not None:
        expected = set(range(args.expected_chains))
        missing = sorted(expected - set(chain_ids))
        extra = sorted(set(chain_ids) - expected)
        if missing or extra:
            raise ValueError(
                f"Shard chain-id mismatch. missing={missing}, extra={extra}, "
                f"found={chain_ids}"
            )

    ref_L = entries[0]["L"]
    ref_beta = entries[0]["beta"]
    for e in entries[1:]:
        if e["L"] != ref_L:
            raise ValueError(f"Inconsistent L: {e['path']} has L={e['L']} != {ref_L}")
        if abs(e["beta"] - ref_beta) > 1e-12:
            raise ValueError(
                f"Inconsistent beta: {e['path']} has beta={e['beta']} != {ref_beta}"
            )

    ops_history = np.concatenate([e["ops_history"] for e in entries], axis=0)
    wilson_history = np.concatenate([e["wilson_history"] for e in entries], axis=0)
    wilson_avg = np.mean(wilson_history, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        ops_history=ops_history,
        wilson_history=wilson_history,
        wilson_avg=wilson_avg,
        beta=ref_beta,
        L=ref_L,
        merged_chain_ids=np.array(chain_ids, dtype=np.int64),
        source_files=np.array([e["path"] for e in entries]),
    )

    print(f"Merged {len(entries)} shards -> {args.output}")
    print(f"Total configs: {ops_history.shape[0]}")
    print(f"L={ref_L}, beta={ref_beta}")

    if args.delete_input_shards:
        removed = 0
        for e in entries:
            os.remove(e["path"])
            removed += 1
        print(f"Deleted {removed} shard files.")


if __name__ == "__main__":
    main()
