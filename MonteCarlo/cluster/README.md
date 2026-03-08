# MonteCarlo Cluster Prep

This folder contains a shard-based Harvard RC workflow for
`MonteCarlo/latticeYM3d-simulation.py`.

## Layout
- `scripts/run_latticeYM3d_shard.py`: runs one chain shard and saves `shard_XXXX.npz`
- `scripts/merge_latticeYM3d_shards.py`: merges shard files into one `lattice_data_3d.npz`-compatible file
- `scripts/merge_latticeYM3d_beta_scan.py`: merges all beta-point shard folders into per-beta merged files
- `slurm/run_latticeYM3d_shard_harvard.slurm`: one Slurm array task = one chain shard
- `slurm/submit_latticeYM3d_harvard.sh`: submits the array job
- `slurm/submit_latticeYM3d_beta_scan_harvard.sh`: submits one array job per beta point
- `env.harvard.example`: environment template
- `env.harvard.1gb.example`: ready profile capped around 1GB merged output
- `env.harvard.beta_scan.example`: beta-scan template
- `sync_to_harvard.sh`: rsync helper for first-time/project updates

## Why Shards
- Chains are independent, so sharding scales across many nodes/cores.
- This avoids multi-node MPI coupling and keeps failure/restart simple.
- Precision scaling is mainly via `TOTAL_MEAS` (more configurations).

## Quick Start (Harvard RC)
0. (Optional) Sync from local machine:
   - `export RC_LOGIN='user@login.rc.fas.harvard.edu'`
   - `export RC_BASE_DIR='/n/home00/user/projects'`
   - `./sync_to_harvard.sh`
1. Copy and edit env:
   - `cp env.harvard.example env.harvard`
   - `source env.harvard`
2. Choose partition:
   - `export PARTITION=shared` or `export PARTITION=yin`
3. Submit:
   - `cd slurm`
   - `./submit_latticeYM3d_harvard.sh`
4. After all shards finish, merge:
   - `python3 ../scripts/merge_latticeYM3d_shards.py --input-dir ../results/harvard_shards --expected-chains "$TOTAL_CHAINS" --output ../results/lattice_data_3d_merged.npz`
   - To save space immediately: add `--delete-input-shards`

## Beta Scan (Harvard RC)
1. Configure scan env:
   - `cp env.harvard.beta_scan.example env.harvard.scan`
   - `source env.harvard.scan`
2. Submit all beta points:
   - `cd slurm`
   - `./submit_latticeYM3d_beta_scan_harvard.sh`
   - This writes a submission manifest to `../results/beta_scan/<scan_tag>/submit_manifest.tsv`.
3. After all scan jobs finish, merge each beta:
   - `python3 ../scripts/merge_latticeYM3d_beta_scan.py --scan-root ../results/beta_scan/<scan_tag> --expected-chains "$TOTAL_CHAINS" --output-root ../results/beta_scan_merged/<scan_tag>`
   - To save space immediately: add `--delete-input-shards`

Notes:
- `BETA_LIST` accepts spaces or commas, e.g. `5.5 6.0 7.0` or `5.5,6.0,7.0`.
- Each beta point gets an isolated shard folder at `.../beta_<tag>/harvard_shards`.
- Output merged files are named `lattice_data_3d_beta_<tag>.npz`.

## Precision Knobs
- `TOTAL_MEAS`: dominant precision control (increase first)
- `N_THERM`: thermalization sweeps per chain
- `N_SKIP`: decorrelation sweeps between measurements
- `TOTAL_CHAINS`: parallelism level (does not change total statistics if `TOTAL_MEAS` fixed)

## Notes
- `PARTITION` is passed at submit time. Use `yin` for dedicated access if available.
- To avoid oversubscription in array jobs, shard jobs default to one BLAS/OpenMP thread.
- Merged output includes `ops_history`, `wilson_history`, and `wilson_avg`, matching analysis expectations.
- Approx storage at `L=32`: about `1056 * TOTAL_MEAS` bytes for merged data (roughly 3.2 MB for `TOTAL_MEAS=3000`).
- For ~1GB cap, start from `env.harvard.1gb.example` (`TOTAL_MEAS=900000`).
- For beta scans, total merged storage scales roughly as `N_beta * 1056 * TOTAL_MEAS` bytes at `L=32`.
