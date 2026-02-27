# Bootstrap Workflows

Production-ready numerical bootstrap workflows live in subfolders.

## Available

- **`ising3d/`** -- Single-correlator 3D Ising bootstrap (Z2-even sector).
  Builds conformal block tables with PyCFTBoot, solves the SDP with SDPB,
  and bisects to find the upper bound on Delta_epsilon.
  Local (Docker) and Harvard RC (SLURM) runners included.

- **`ising3d_mixed/`** -- Mixed-correlator 3D Ising bootstrap island scan.
  Scans a (Delta_sigma, Delta_epsilon) grid using the full mixed-correlator
  sum rule (5 matrix + 2 vector channels) to map out the allowed island.
  Local and Harvard RC runners included.

## Quick Start

Single-correlator bound at one point:
```bash
cd Bootstrap/ising3d
source setup_sdpb_local.sh   # Docker-backed SDPB wrappers
bash run_point_local.sh 0.518
```

Mixed-correlator island scan (local):
```bash
cd Bootstrap/ising3d_mixed
source ../ising3d/setup_sdpb_local.sh
bash run_island_local.sh
```

See the README in each subfolder for full details and cluster submission instructions.
