#!/bin/sh
# Direct-displacement (u,v,w) PINN, 20% spatial / 20% temporal -- same hyperparameters
# as the potential-PINN baseline potential_pinn/outputs/sp20_t20_silu_F256_h256x256x256_a1_hpc
# (run_grid.sbatch). Only the model output + physics loss differ.
# Submit FROM the revision `code/` folder (the one holding potential_pinn/ and direct_uvw_pinn/):
#   sbatch direct_uvw_pinn/run_uvw_sp20_t20.sbatch
# Override the dataset / python with --export=ALL,CSV=...,PY=...
# The CSV's *_meta.json must sit next to it (gives the "sp20" tag).
#SBATCH --job-name=uvw20
#SBATCH -N 1
#SBATCH -p gpu-v100-32gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p logs
unset PYTORCH_CUDA_ALLOC_CONF
PY=${PY:-/work/nmehtaj/envs/pinn/bin/python}
CSV=${CSV:-potential_pinn/data/dataset_layerwise_20pct_3ply_fullsignal_6001steps.csv}
MAT=${MAT:-data/3D_Pristine.mat}
hostname; date; echo "CSV=$CSV PY=$PY"

$PY -u direct_uvw_pinn/scripts/run_exp_uvw.py \
    --csv "$CSV" --temporal 0.20 \
    --activation silu --num-freq 256 --hidden 256 256 256 \
    --balance-alpha 1.0 --epochs 600 --patience 50 \
    --data-only-epochs 30 --bdry 1 --dtype float64 \
    --batch-size 16384 --seed 42 --n-held-spatial 10 \
    --ckpt-every 5 --resume --tag-suffix _uvw_hpc

OUT=direct_uvw_pinn/outputs/sp20_t20_silu_F256_h256x256x256_a1_uvw_hpc
# full-field snapshots at the same timesteps as the baseline (needs the .mat)
[ -f "$MAT" ] && $PY -u direct_uvw_pinn/scripts/reconstruct_field_uvw.py --out "$OUT" \
    --mat "$MAT" --timesteps 300 3000 5800 | tee "$OUT/field_rel_l2.log"
$PY -u compare.py
date
