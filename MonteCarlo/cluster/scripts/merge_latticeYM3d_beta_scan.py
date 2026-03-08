#!/usr/bin/env python3
"""
Merge all beta-scan shard outputs into per-beta merged npz files.

Expected layout:
  <scan_root>/beta_<tag>/harvard_shards/shard_XXXX.npz
or
  <scan_root>/beta_<tag>/shard_XXXX.npz
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys


def parse_beta_list(beta_list):
    tokens = beta_list.replace(",", " ").split()
    if not tokens:
        raise ValueError("BETA_LIST is empty.")
    values = []
    for tok in tokens:
        values.append(float(tok))
    return values


def beta_to_tag(beta):
    s = str(beta)
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def find_beta_dirs(scan_root):
    out = []
    for path in sorted(glob.glob(os.path.join(scan_root, "beta_*"))):
        if os.path.isdir(path):
            out.append(path)
    return out


def pick_input_dir(beta_dir):
    preferred = os.path.join(beta_dir, "harvard_shards")
    if os.path.isdir(preferred):
        return preferred
    return beta_dir


def main():
    parser = argparse.ArgumentParser(description="Merge all beta points from a scan root.")
    default_scan_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "beta_scan",
    )
    default_output_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "beta_scan_merged",
    )
    default_merge_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "merge_latticeYM3d_shards.py",
    )
    parser.add_argument("--scan-root", default=default_scan_root)
    parser.add_argument(
        "--beta-list",
        default=None,
        help="Optional beta list, e.g. '5.5 6.0 7.0' or '5.5,6.0,7.0'.",
    )
    parser.add_argument("--expected-chains", type=int, default=None)
    parser.add_argument("--output-root", default=default_output_root)
    parser.add_argument("--merge-script", default=default_merge_script)
    parser.add_argument(
        "--delete-input-shards",
        action="store_true",
        help="Delete shards after each successful per-beta merge.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.merge_script):
        raise FileNotFoundError(f"Merge script not found: {args.merge_script}")
    if not os.path.isdir(args.scan_root):
        raise FileNotFoundError(f"Scan root not found: {args.scan_root}")
    os.makedirs(args.output_root, exist_ok=True)

    if args.beta_list is not None:
        betas = parse_beta_list(args.beta_list)
        beta_dirs = [
            os.path.join(args.scan_root, f"beta_{beta_to_tag(beta)}")
            for beta in betas
        ]
    else:
        beta_dirs = find_beta_dirs(args.scan_root)

    if not beta_dirs:
        raise FileNotFoundError(f"No beta directories found under {args.scan_root}")

    merged = []
    for beta_dir in beta_dirs:
        if not os.path.isdir(beta_dir):
            raise FileNotFoundError(f"Beta directory not found: {beta_dir}")

        beta_tag = os.path.basename(beta_dir).replace("beta_", "", 1)
        input_dir = pick_input_dir(beta_dir)
        out_file = os.path.join(args.output_root, f"lattice_data_3d_beta_{beta_tag}.npz")

        cmd = [
            sys.executable,
            args.merge_script,
            "--input-dir",
            input_dir,
            "--output",
            out_file,
        ]
        if args.expected_chains is not None:
            cmd.extend(["--expected-chains", str(args.expected_chains)])
        if args.delete_input_shards:
            cmd.append("--delete-input-shards")

        print("Running:", " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)
        merged.append((beta_tag, out_file))

    manifest_path = os.path.join(args.output_root, "merge_manifest.tsv")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.write("beta_tag\toutput_file\n")
        for beta_tag, out_file in merged:
            fh.write(f"{beta_tag}\t{out_file}\n")

    print(f"Merged {len(merged)} beta points.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
