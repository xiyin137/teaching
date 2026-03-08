# First-Time Harvard RC Deployment (MonteCarlo)

Use this when your cluster account has no MonteCarlo files yet.

## 1. Set local deployment vars
```bash
export RC_LOGIN='your_harvard_rc_username@login.rc.fas.harvard.edu'
export RC_BASE_DIR='/n/home00/your_harvard_rc_username/projects'
```

## 2. Create remote project directory
```bash
ssh "$RC_LOGIN" "mkdir -p '$RC_BASE_DIR/MonteCarlo/cluster'"
```

## 3. Sync project files
From local repo root (`/Users/xiyin/teaching`):
```bash
./MonteCarlo/cluster/sync_to_harvard.sh
```

This syncs:
- `MonteCarlo/latticeYM3d-simulation.py`
- `MonteCarlo/latticeYM3d-analysis.py`
- `MonteCarlo/latticeYM3d-errors.py`
- `MonteCarlo/cluster/` (scripts, slurm files, docs)

## 4. Remote environment setup (once)
```bash
ssh "$RC_LOGIN"
cd "$RC_BASE_DIR/MonteCarlo"
module purge
module load python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy numba matplotlib
```

## 5. Configure and submit
```bash
cd "$RC_BASE_DIR/MonteCarlo/cluster"
cp env.harvard.example env.harvard
# edit env.harvard for partition/settings (shared vs yin)
source env.harvard
cd slurm
./submit_latticeYM3d_harvard.sh
```

## 6. Merge outputs after jobs finish
```bash
cd "$RC_BASE_DIR/MonteCarlo/cluster/slurm"
python3 ../scripts/merge_latticeYM3d_shards.py \
  --input-dir ../results/harvard_shards \
  --expected-chains "$TOTAL_CHAINS" \
  --output ../results/lattice_data_3d_merged.npz \
  --delete-input-shards
```

## 7. Optional: run analysis
```bash
cd "$RC_BASE_DIR/MonteCarlo"
python3 latticeYM3d-analysis.py
python3 latticeYM3d-errors.py
```

## 8. Optional: submit a beta scan
```bash
cd "$RC_BASE_DIR/MonteCarlo/cluster"
cp env.harvard.beta_scan.example env.harvard.scan
source env.harvard.scan
cd slurm
./submit_latticeYM3d_beta_scan_harvard.sh
```

After scan jobs finish, merge each beta:
```bash
cd "$RC_BASE_DIR/MonteCarlo/cluster/slurm"
python3 ../scripts/merge_latticeYM3d_beta_scan.py \
  --scan-root ../results/beta_scan/<scan_tag> \
  --expected-chains "$TOTAL_CHAINS" \
  --output-root ../results/beta_scan_merged/<scan_tag> \
  --delete-input-shards
```
