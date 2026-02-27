# ConformalBlocks

Streamlined demo scripts for conformal blocks and the conformal bootstrap.

## Main End-to-End Demo
- `conformal_blocks_and_bootstrap_demo_streamlined.py`
- `Conformal blocks and bootstrap demo streamlined.wl`
- `LiouvilleCFT_streamlined.nb`
- `ConformalBlock.m`

`LiouvilleCFT_streamlined.nb` keeps original-faithful numeric settings for Upsilon/DOZZ (`prec=100`, `MaxRecursion->1000`, `TimeConstraint->1000`), so runs can be significantly slower but more stable.

## Focused Demos
- `2D_conformal_blocks_demo_streamlined.py`
- `2D_conformal_blocks_demo_streamlined.wl`

3D Ising bootstrap production scripts live in `Bootstrap/ising3d/`.

## Python Dependencies
```bash
cd ConformalBlocks
python3 -m pip install -r requirements.txt
```

For `mode=3d`, you also need SDPB/PMP executables (or Docker wrappers via `Bootstrap/ising3d/setup_sdpb_local.sh`).

## Quick Start
```bash
cd ConformalBlocks
./conformal_blocks_and_bootstrap_demo_streamlined.py 2d
./conformal_blocks_and_bootstrap_demo_streamlined.py 4d
```

For 3D SDPB runs, use either:
- `./conformal_blocks_and_bootstrap_demo_streamlined.py 3d`
- or the full workflow in `Bootstrap/ising3d`

Note: local 3D defaults use `cutoff=0.20` for better stability at higher truncation.

For local Docker-backed SDPB before `mode=3d`:
```bash
cd Bootstrap/ising3d
bash setup_python_local.sh
source setup_sdpb_local.sh
cd ConformalBlocks
./conformal_blocks_and_bootstrap_demo_streamlined.py 3d
```

Convenience launcher (does the setup step automatically):
```bash
cd ConformalBlocks
./run_3d_demo_local.sh --delta-sigma 0.518 --lower 1.1 --upper 1.3 --tol 0.05
```

