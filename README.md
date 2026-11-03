# PICS: Physics-Informed Compressive Sensing for Sparse Wavefield Reconstruction

## Overview

PICS combines sparse sensing with a Physics-Informed Neural Network (PINN) to reconstruct the
full three-dimensional guided-wave displacement field $(u, v, w)$ of a plate from a small
fraction of its material points and time samples. Instead of relying on sparsity priors and
optimisation-based recovery, the network learns four elastodynamic potentials
$(\phi, \psi_x, \psi_y, \psi_z)$ and obtains the displacement through the Helmholtz
decomposition $\mathbf{u} = \nabla\phi + \nabla\times\boldsymbol{\psi}$, so every prediction
is tied to the wave equations, the Coulomb gauge, a rest initial condition and the plate
boundary condition through the loss.

The framework is evaluated on a 300 x 200 x 2 mm aluminium plate excited by a 150 kHz tone
burst, with training data retained at **1, 10 and 20 % of the spatial points** crossed with
**1, 10 and 20 % of the time samples** (nine models). At 20 % / 20 % the model reconstructs
unseen points with a median relative-L2 error of 0.078 and infills unseen time samples at
seen points with 0.041.

**Associated Publication:**
*Physics-Informed Compressive Sensing Using PINNs for Sparse Wavefield Reconstruction in
Digital Twin-Enabled Structural Health Monitoring*
Integrated Material Assessment and Predictive Simulation Laboratory (i-MAPS),
University of South Carolina

---

## Problem Context

Continuous monitoring of large aerospace, naval and defence structures produces far more
guided-wave data than can be stored or transmitted across edge, fog and cloud tiers. PICS
transmits only a sparse subset of sensor points and time samples and reconstructs the complete
wavefield on the receiving side, using the governing physics to fill in what was never sent.
The ground truth is a peri-elastodynamic (PED) simulation of the plate: 180,000 material points
(three through-thickness plies at z = 0, -1, -2 mm) recorded for 6001 steps at 10 ns.

---

## Repository Structure

```
├── Codes/
│   ├── forward/                      # PINN reconstruction (the nine Fig. 8 models)
│   │   ├── scinonet_pinn.py          #   model, Fourier features, physics losses, training, plots
│   │   ├── gen_layerwise_dataset.py  #   sample the .mat into a training CSV
│   │   ├── run_exp.py                #   train + evaluate one configuration
│   │   ├── eval_from_ckpt.py         #   evaluate a run that stopped at the wall limit
│   │   ├── reconstruct_field.py      #   full top-ply field at chosen timesteps
│   │   ├── plot_plate.py             #   plate-layout figure of a run's sampling
│   │   ├── make_results_table.py     #   seen / unseen error table
│   │   ├── make_run_cards.py         #   README card per run + catalog table below
│   │   ├── run_search.py             #   activation / alpha / architecture search
│   │   └── run_phys_sweep.py         #   physics-weight sweep
│   └── figures/                      # manuscript figure scripts
│       ├── regen_fig6_extrap_interp.py     # Fig. 6
│       ├── regen_pred_u_3x3_grid.py        # Fig. 8
│       ├── regen_source_to_boundary.py     # Fig. 11
│       ├── regen_fig12_layers.py           # Fig. 12a-c
│       └── regen_fig12_viridis.py          # Fig. 12a-c, viridis style
├── Dataset/
│   ├── 3D_Pristine.mat               # (not tracked) 16 GB PED simulation, MATLAB v7.3
│   └── dataset_layerwise_*_meta.json # provenance of the generated training CSVs (CSVs not tracked)
├── Results/
│   ├── forward_runs/<tag>/           # nine trained models: model.pt, config.json, metrics.json, figures
│   └── figures/                      # every manuscript figure (PNG + SVG) and the static schematics
├── default_config.yaml               # hyperparameters and paths in one place
├── CHANGELOG.md                      # version history and script-to-figure mapping
├── FILE_RENAME_REFERENCE.md          # where each file came from
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

`Codes/forward/scinonet_pinn.py` is the PICS forward model. The filename is historical and is
kept because the saved checkpoints reference it when they are loaded.

---

### Script Descriptions

#### 1. Dataset Generation

`Codes/forward/gen_layerwise_dataset.py --pct {1,10,20}` draws that percentage of the 60,000
first-ply (x, y) grid points, takes each at all three plies, and writes the full 6001-step
signals in long format `[x, y, z, t, u, v, w, is_boundary]` (mm, s, mm). Twenty percent of
the budget is forced onto the plate perimeter and flagged `is_boundary = 1`; these points are
never spatially held out. Temporal subsampling is applied at training time by `--temporal`.

#### 2. Training

`Codes/forward/run_exp.py` trains one (spatial %, temporal %) cell. The network maps
$(x, y, z, t)$ through dispersion-clamped Fourier features and a SiLU backbone
(256 x 256 x 256) to the four potentials. Loss terms:

| Term | Residual | Points per step |
|---|---|---|
| data | $\mathbf{u}_{pred} - \mathbf{u}_{obs}$ at kept samples | batch 16384 |
| wave | $c^2 \nabla^2 q - q_{tt}$ for $\phi$ (P) and $\boldsymbol{\psi}$ (S) | 2048 collocation |
| gauge | $\nabla\cdot\boldsymbol{\psi}$ | same |
| initial | potentials and their time derivatives at $t = 0$ | 1024 |
| boundary | $u = v = w = 0$ on the four plate edges | 1024 |

Physics is switched on after a 30-epoch data-only warmup, and its weight is set every epoch by
gradient-norm balancing (`balance_alpha = 1.0`). All runs use float64, Adam (lr 2e-3),
ReduceLROnPlateau and gradient clipping at 1.0. Each run writes `model.pt`, `metrics.json`,
`config.json`, `progress.csv`, the loss curves and the seen / unseen reconstructions.

#### 3. Inference and Figures

| Script | Output |
|---|---|
| `Codes/figures/regen_fig6_extrap_interp.py` | Fig. 6: seen vs unseen point signals, 10 % spatial, temporal 1 / 10 / 20 % |
| `Codes/figures/regen_pred_u_3x3_grid.py` | Fig. 8: predicted $u$ on the top ply at 58.0 us for the nine models |
| `Codes/figures/regen_source_to_boundary.py` | Fig. 11: four full-signal predictions from source to boundary, plus layout |
| `Codes/figures/regen_fig12_layers.py`, `regen_fig12_viridis.py` | Fig. 12a-c: three-layer u / v / w reconstruction, two colour styles |
| `Codes/forward/reconstruct_field.py` | full top-ply field at chosen timesteps for one run |
| `Codes/forward/make_results_table.py` | Table 3 rows for one run |

---

## Architecture Summary

- **Input:** standardised $(x, y, z, t)$; per-axis lengths and a time scale $\tau = L_x / c_p$
  make the P-wave coefficient 1 and the S-wave coefficient $\gamma^2 = (c_s / c_p)^2$.
- **Fourier features:** 256 learnable frequencies, spatial band clamped to 0.13 cyc/mm x 2.0
  and temporal band to 300 kHz from the Lamb dispersion curves; raw coordinates concatenated.
- **Backbone:** three hidden layers of 256, SiLU.
- **Head:** four potentials. Displacement and all residuals are autograd derivatives of them
  (second-order, `create_graph=True`), which is why training uses float64.

---

## Experimental Setup

| Item | Value |
|---|---|
| Plate | 300 x 200 x 2 mm, aluminium 6061-T6, $c_p$ = 6.3 mm/us, $c_s$ = 3.2 mm/us |
| Source data | PED simulation, 1 mm grid, 3 plies, 6001 steps at 10 ns (0-60 us) |
| Excitation | 3.5-cycle 150 kHz tone burst at (-50.5, -99.5) mm |
| Spatial coverage | 1, 10, 20 % of first-ply points, all plies, 20 % of budget on the perimeter |
| Temporal coverage | 1, 10, 20 % of time samples per trained point |
| Held-out test | 10 unseen spatial points (extrapolation); held-out time samples at seen points (interpolation) |
| Hardware | 1 x NVIDIA V100 32 GB; 600 epochs is roughly 48 h at 20 % spatial |

---

## Key Results

Median relative-L2 error from each run's `metrics.json` (`unseen` = extrapolation at points
never seen, `seen` = interpolation of held-out time samples at trained points):

| run tag | data | temporal | activation | F | hidden | alpha | unseen | seen |
|---|---|---|---|---|---|---|---|---|
| sp1_t1_silu_F256_h256x256x256_a1_hpc | 1.0% | 1% | silu | 256 | 256x256x256 | 1.0 | 1.040 | 0.427 |
| sp1_t10_silu_F256_h256x256x256_a1_hpc | 1.0% | 10% | silu | 256 | 256x256x256 | 1.0 | 1.179 | 0.140 |
| sp1_t20_silu_F256_h256x256x256_a1_hpc | 1.0% | 20% | silu | 256 | 256x256x256 | 1.0 | 1.213 | 0.086 |
| sp10_t1_silu_F256_h256x256x256_a1_hpc | 10.0% | 1% | silu | 256 | 256x256x256 | 1.0 | 0.319 | 0.180 |
| sp10_t10_silu_F256_h256x256x256_a1_hpc | 10.0% | 10% | silu | 256 | 256x256x256 | 1.0 | 0.229 | 0.090 |
| sp10_t20_silu_F256_h256x256x256_a1_hpc | 10.0% | 20% | silu | 256 | 256x256x256 | 1.0 | 0.314 | 0.078 |
| sp20_t1_silu_F256_h256x256x256_a1_hpc | 20.0% | 1% | silu | 256 | 256x256x256 | 1.0 | 0.153 | 0.136 |
| sp20_t10_silu_F256_h256x256x256_a1_hpc | 20.0% | 10% | silu | 256 | 256x256x256 | 1.0 | 0.089 | 0.046 |
| sp20_t20_silu_F256_h256x256x256_a1_hpc | 20.0% | 20% | silu | 256 | 256x256x256 | 1.0 | 0.078 | 0.041 |

Field relative-L2 of the predicted $u$ at t = 58.0 us (the numbers printed on Fig. 8):
1 % spatial 1.11 / 1.53 / 1.42, 10 % spatial 0.83 / 0.75 / 0.68, 20 % spatial 0.29 / 0.17 / 0.18
for temporal 1 / 10 / 20 %.

---

## Dependencies

- Python 3.10+
- PyTorch 2.2+ (CUDA recommended for training; the figure scripts work on CPU)
- NumPy, Pandas, Matplotlib
- h5py (reads the MATLAB v7.3 simulation file)
- Pillow (Fig. 6)

```bash
pip install -r requirements.txt
```

---

## Usage

Run the forward scripts from the repository root (their defaults are relative to it).
The figure scripts resolve paths from their own location and run from anywhere.
Put the simulation file at `Dataset/3D_Pristine.mat` first.

### Dataset Preparation

```bash
python Codes/forward/gen_layerwise_dataset.py --pct 20      # -> Dataset/dataset_layerwise_20pct_3ply_fullsignal_6001steps.csv
```

### Training (20 % spatial / 20 % temporal, the headline run)

```bash
python Codes/forward/run_exp.py \
    --csv Dataset/dataset_layerwise_20pct_3ply_fullsignal_6001steps.csv \
    --activation silu --num-freq 256 --hidden 256 256 256 \
    --balance-alpha 1.0 --temporal 0.20 --data-only-epochs 30 \
    --epochs 600 --patience 50 --dtype float64 --ckpt-every 5 --resume --tag-suffix _hpc
```

Writes `Results/forward_runs/sp20_t20_silu_F256_h256x256x256_a1_hpc/`. Repeat with `--pct`
and `--temporal` in {1, 10, 20} % for the other eight cells.

### Figures

```bash
python Codes/figures/regen_pred_u_3x3_grid.py          # Fig. 8 (reads the nine model.pt files)
python Codes/figures/regen_fig6_extrap_interp.py       # Fig. 6
python Codes/figures/regen_source_to_boundary.py       # Fig. 11
python Codes/figures/regen_fig12_layers.py             # Fig. 12a-c
```

Each writes PNG/SVG next to the script. The scripts that need the full plate read
`Dataset/3D_Pristine.mat` directly and cache their inference in a `.npz` (git-ignored); pass
`--refresh` to recompute.

---

## Notes

- The simulation file, the training CSVs and the figure caches are not tracked (see
  `.gitignore`); everything needed to regenerate the figures from the trained models is in
  `Results/forward_runs/`.
- `sp1_t10` and `sp1_t20` were trained for 80 epochs, the other seven for 600 (`sp20_t10`
  640, stopped at the wall limit and evaluated from its checkpoint).
- `Codes/forward/make_run_cards.py` rewrites the catalog table above from the run folders.
- The scripts that originally produced the Fig. 7 sweep plots were not recovered; the images
  are in `Results/figures/`.

---

## File Reference

Files keep their original names; `FILE_RENAME_REFERENCE.md` lists where each came from.

---

## Citation

If you use this code or methodology, please cite:

```
TBD
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Sourav Banerjee (Corresponding Author) - banerjes@cec.sc.edu
Integrated Material Assessment and Predictive Simulation Laboratory (i-MAPS)
Department of Mechanical Engineering, University of South Carolina
