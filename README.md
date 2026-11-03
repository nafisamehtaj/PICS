# PICS: Physics-Informed Compressive Sensing for Sparse Wavefield Reconstruction

Reconstructs the full 3D guided-wave displacement field $(u, v, w)$ of a plate from 1 to 20 %
of its material points and 1 to 20 % of its time samples, using a Physics-Informed Neural
Network (PINN) built on Helmholtz potentials. Code and results for the manuscript
*Physics-Informed Compressive Sensing Using PINNs for Sparse Wavefield Reconstruction in
Digital Twin-Enabled Structural Health Monitoring* (i-MAPS, University of South Carolina).

---

## Where is what

| Looking for | Go to |
|---|---|
| **Training configuration** (every hyperparameter, with the reason for each) | [`configs/training_config.yaml`](configs/training_config.yaml) |
| The exact command for each of the nine trained models | [`configs/train_all_nine.sh`](configs/train_all_nine.sh) |
| What a given model was actually trained with | `Results/forward_runs/<run>/config.json` |
| **Dataset generation** (sample the simulation into a training CSV) | [`Codes/step1_generate_dataset.py`](Codes/step1_generate_dataset.py) |
| Train / validation / held-out split, temporal subsampling | [`Codes/pics_pinn.py` L270-L318](Codes/pics_pinn.py#L270-L318) `build_dataset` |
| Non-dimensionalisation (scalers, $\beta_y$, $\beta_z$, $\gamma$) | [`Codes/pics_pinn.py` L198-L240](Codes/pics_pinn.py#L198-L240) `PotentialScalers` |
| Fourier-feature embedding | [`Codes/pics_pinn.py` L390-L422](Codes/pics_pinn.py#L390-L422) `SpecializedFourierFeatures` |
| **Network** (backbone + four-potential head) | [`Codes/pics_pinn.py` L434-L464](Codes/pics_pinn.py#L434-L464) `PotentialNet` |
| Helmholtz displacement $\mathbf{u} = \nabla\phi + \nabla\times\boldsymbol{\psi}$ | [`Codes/pics_pinn.py` L471-L479](Codes/pics_pinn.py#L471-L479) `displacement` |
| **Physics loss: wave equation residual** | [`Codes/pics_pinn.py` L481-L502](Codes/pics_pinn.py#L481-L502) `_wave_residual`, `physics_residuals` |
| Physics loss: initial condition | [`Codes/pics_pinn.py` L504-L508](Codes/pics_pinn.py#L504-L508) `ic_residual` |
| Physics loss: plate boundary condition | [`Codes/pics_pinn.py` L510-L541](Codes/pics_pinn.py#L510-L541) `boundary_residual` |
| **Training loop**, total loss, gradient-norm balancing, early stopping | [`Codes/pics_pinn.py` L557-L776](Codes/pics_pinn.py#L557-L776) `train` |
| **Train one model** (command-line entry point) | [`Codes/step2_train.py`](Codes/step2_train.py) |
| Evaluate seen / unseen error from a checkpoint | [`Codes/step3_evaluate_checkpoint.py`](Codes/step3_evaluate_checkpoint.py) |
| Predict the whole plate at chosen times | [`Codes/step4_reconstruct_field.py`](Codes/step4_reconstruct_field.py) |
| Results table for one model | [`Codes/step5_results_table.py`](Codes/step5_results_table.py) |
| **Manuscript figures** | `Codes/figures/fig06_*.py`, `fig08_*.py`, `fig11_*.py`, `fig12_*.py` |
| Trained models, their metrics and loss curves | `Results/forward_runs/` (nine runs) |
| All figure images (PNG + SVG) | `Results/figures/` |

---

## The physics in the loss

The network outputs four potentials $(\phi, \psi_x, \psi_y, \psi_z)$ of the non-dimensional
coordinates $(x, y, z, t)$. Everything else is derived from them by automatic differentiation.

| Term | Equation enforced | Where |
|---|---|---|
| Data | $\mathbf{u}_{pred}(x,y,z,t) = \mathbf{u}_{obs}$ at the kept samples | `train` |
| P-wave | $\nabla^2\phi - \phi_{tt} = 0$ | `_wave_residual` with `coeff2 = 1` |
| S-wave | $\gamma^2\nabla^2\boldsymbol{\psi} - \boldsymbol{\psi}_{tt} = 0$, $\gamma = c_s/c_p$ | `_wave_residual` with `coeff2 = gamma^2` |
| Gauge | $\nabla\cdot\boldsymbol{\psi} = 0$ (Coulomb) | `physics_residuals` |
| Initial | $\phi = \boldsymbol{\psi} = \phi_t = \boldsymbol{\psi}_t = 0$ at $t = 0$ | `ic_residual` |
| Boundary | $u = v = w = 0$ on the four plate edges | `boundary_residual` |

with the scaled Laplacian $\nabla^2 q = q_{xx} + \beta_y^2 q_{yy} + \beta_z^2 q_{zz}$ and
$\mathbf{u} = \nabla\phi + \nabla\times\boldsymbol{\psi}$. All terms are mean-squared residuals.

Total loss:

```
epochs 1..30          L = L_data                                         (data-only warm-up)
after warm-up         L = L_data + lambda * (L_wave + L_gauge + L_ic + L_bdry)
```

`lambda` is not a fixed number. Once per epoch it is set so that the physics gradient norm is
`balance_alpha` (= 1.0) times the data gradient norm, smoothed with an EMA of 0.9. The code is
in `train`, [`Codes/pics_pinn.py` L557-L776](Codes/pics_pinn.py#L557-L776).

---

## Repository layout

```
├── Codes/
│   ├── pics_pinn.py                  # the library: data split, Fourier features, network,
│   │                                 #   physics residuals, loss, training loop, plot styles
│   ├── step1_generate_dataset.py     # 3D_Pristine.mat  ->  Dataset/<pct>.csv
│   ├── step2_train.py                # Dataset/<pct>.csv -> Results/forward_runs/<run>/
│   ├── step3_evaluate_checkpoint.py  # re-evaluate a run from checkpoint.pt
│   ├── step4_reconstruct_field.py    # whole-plate prediction at chosen timesteps
│   ├── step5_results_table.py        # seen / unseen error table for one run
│   ├── figures/                      # one script per manuscript figure
│   │   ├── fig06_extrap_interp.py          # Fig. 6
│   │   ├── fig08_coverage_grid.py          # Fig. 8
│   │   ├── fig11_source_to_boundary.py     # Fig. 11
│   │   ├── fig12_layers.py                 # Fig. 12a-c
│   │   └── fig12_viridis.py                # Fig. 12a-c, viridis style
│   └── tools/                        # plate-layout plot, run cards, config search, alpha sweep
├── configs/
│   ├── training_config.yaml          # all hyperparameters in one place
│   └── train_all_nine.sh             # the nine training commands
├── Dataset/
│   ├── 3D_Pristine.mat               # (not tracked) 16 GB PED simulation, MATLAB v7.3
│   └── dataset_layerwise_*_meta.json # provenance of the generated CSVs (CSVs not tracked)
├── Results/
│   ├── forward_runs/<run>/           # model.pt, config.json, metrics.json, loss curves, plots
│   └── figures/                      # every manuscript figure (PNG + SVG)
├── CHANGELOG.md                      # version history and script-to-figure mapping
├── FILE_RENAME_REFERENCE.md          # where each file came from
├── requirements.txt
└── LICENSE                           # MIT
```

---

## Quick start

```bash
pip install -r requirements.txt
# put the simulation file at Dataset/3D_Pristine.mat, then from the repository root:

python Codes/step1_generate_dataset.py --pct 20
python Codes/step2_train.py \
    --csv Dataset/dataset_layerwise_20pct_3ply_fullsignal_6001steps.csv \
    --temporal 0.20 --activation silu --num-freq 256 --hidden 256 256 256 \
    --balance-alpha 1.0 --data-only-epochs 30 --epochs 600 --patience 50 \
    --dtype float64 --ckpt-every 5 --resume --tag-suffix _hpc
```

That is the headline 20 % spatial / 20 % temporal model. `configs/train_all_nine.sh` runs all
nine. Each run lands in `Results/forward_runs/<run>/` with `model.pt`, `config.json`,
`metrics.json`, `progress.csv`, the loss curves and the seen / unseen reconstruction plots.

The figure scripts need only the shipped `model.pt` files (and `Dataset/3D_Pristine.mat` for
the full-plate figures):

```bash
python Codes/figures/fig08_coverage_grid.py        # Fig. 8
python Codes/figures/fig06_extrap_interp.py        # Fig. 6
python Codes/figures/fig11_source_to_boundary.py   # Fig. 11
python Codes/figures/fig12_layers.py               # Fig. 12a-c
```

They write PNG/SVG next to the script and cache the slow inference in a `.npz`; pass
`--refresh` to recompute.

---

## Data

| Item | Value |
|---|---|
| Plate | 300 x 200 x 2 mm, aluminium 6061-T6, $c_p$ = 6.3 mm/us, $c_s$ = 3.2 mm/us |
| Simulation | peri-elastodynamic, 1 mm grid, 3 plies (z = 0, -1, -2 mm), 6001 steps at 10 ns |
| Excitation | 3.5-cycle 150 kHz tone burst at (-50.5, -99.5) mm |
| CSV columns | `x, y, z, t, u, v, w, is_boundary` (mm, s, mm) |
| Spatial coverage | 1, 10, 20 % of the 60,000 first-ply points, each taken at all 3 plies; 20 % of the budget on the perimeter |
| Temporal coverage | 1, 10, 20 % of the 6001 steps per trained point, drawn at training time |
| Held out | 10 unseen spatial points (extrapolation); the dropped time samples at seen points (interpolation) |

---

## Results

Median relative-L2 error from each run's `metrics.json`:

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

`unseen` = extrapolation at points never seen; `seen` = interpolation of dropped time samples
at trained points. Field relative-L2 of $u$ at t = 58.0 us, as printed on Fig. 8:
1 % spatial 1.11 / 1.53 / 1.42, 10 % 0.83 / 0.75 / 0.68, 20 % 0.29 / 0.17 / 0.18 for
temporal 1 / 10 / 20 %.

---

## Notes

- The 16 GB simulation, the training CSVs and the `.npz` figure caches are not tracked.
  Everything needed to regenerate the figures from the trained models is in the repository.
- `sp1_t10` and `sp1_t20` were trained for 80 epochs, the other seven for 600 (`sp20_t10`
  640, stopped at the wall limit and evaluated from its checkpoint).
- `Codes/tools/make_run_cards.py` rewrites the results table above from the run folders.
- The script that drew the Fig. 7 sweep plots was not recovered; the images are in
  `Results/figures/`.
- Requirements: Python 3.10+, PyTorch 2.2+, NumPy, Pandas, Matplotlib, h5py, Pillow.
  Training was done on one NVIDIA V100 in float64; the figure scripts run on CPU.

---

## Citation

```
TBD
```

## License

MIT, see [LICENSE](LICENSE).

## Contact

Sourav Banerjee (Corresponding Author) - banerjes@cec.sc.edu
Integrated Material Assessment and Predictive Simulation Laboratory (i-MAPS)
Department of Mechanical Engineering, University of South Carolina
