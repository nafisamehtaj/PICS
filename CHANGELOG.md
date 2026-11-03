# Changelog

All notable changes and experiment mappings for the PICS project are documented here.

---

## Script-to-Figure Mapping

| Script | Manuscript element | What it produces |
|---|---|---|
| `Codes/forward/gen_layerwise_dataset.py` | Fig. 4, Section 2 | Training CSV for a spatial percentage (all 3 plies, perimeter points flagged) |
| `Codes/forward/run_exp.py` + `scinonet_pinn.py` | Table 2, Table 3, Fig. 10 | One trained model per (spatial, temporal) cell: `model.pt`, `metrics.json`, loss curves, seen/unseen reconstructions |
| `Codes/forward/reconstruct_field.py` | - | Full top-ply field at chosen timesteps for one run |
| `Codes/forward/make_results_table.py` | Table 3 | Seen/unseen relative-L2 table for one run |
| `Codes/figures/regen_fig6_extrap_interp.py` | Fig. 6 | Seen vs unseen point signals at 10 % spatial, temporal 1/10/20 % |
| `Codes/figures/regen_pred_u_3x3_grid.py` | Fig. 8 | 3x3 predicted-u grid at t = 58.0 us, nine models |
| `Codes/figures/regen_source_to_boundary.py` | Fig. 11 (i), (ii) | Four-point source-to-boundary signals and the layout schematic |
| `Codes/figures/regen_fig12_layers.py` | Fig. 12a-c | Three-layer full-plate u/v/w reconstruction (RdBu style) |
| `Codes/figures/regen_fig12_viridis.py` | Fig. 12a-c | Same, in the viridis style of the submitted draft |

Static manuscript graphics (Fig. 5 architecture, Fourier-feature schematic, Fig. 2/3 reworks) live in
`Results/figures/` without a generating script. The Fig. 7 sweep plots
(`accuracy_vs_sampling`, `error_vs_temporal_per_spatial`, `unseen_error_vs_temporal`) are included
as images; the script that drew them from the nine `metrics.json` files was not recovered.

---

## Version History

### v0.1.0 - Initial Release (2026-11-03)

**Scope.** The code and results behind the manuscript
*Physics-Informed Compressive Sensing Using PINNs for Sparse Wavefield Reconstruction*:
the forward reconstruction model, the nine spatial x temporal coverage runs, and the scripts
for the figures that appear in the paper. Related work on material identification (inverse
PICS) and the SciNoNet multi-head PINN is kept in separate repositories.

**Organised from the working folders** `scinonet-pinn/experiments/exp2_layerwise` and
`Manuscript Preparation/generated_elements_4july` into a `Codes / Dataset / Results` layout.
See `FILE_RENAME_REFERENCE.md` for where every file came from.

**Path changes only** - no functional change to any model, loss or figure:
- Every script resolves `Dataset/3D_Pristine.mat`, `Results/forward_runs/<tag>/model.pt`
  and `Codes/forward/scinonet_pinn.py` relative to its own location.

**Included artefacts**
- Nine trained forward models (`model.pt`, about 3 MB each) with `config.json`, `metrics.json`,
  `progress.csv` and run figures, for every (1, 10, 20 %) x (1, 10, 20 %) cell.
- All manuscript figure PNG/SVG files and the dataset provenance files.

**Deliberately left out**
- `3D_Pristine.mat` (16 GB), the generated training CSVs and the `.npz` figure caches:
  all regenerable, all in `.gitignore`.

**Known issues**
- `config.json` inside each run folder still records the original dataset path it was trained from.
- Forward scripts with command-line defaults (`gen_layerwise_dataset.py`, `reconstruct_field.py`)
  expect to be run from the repository root; the figure scripts run from anywhere.
- `sp1_t10` and `sp1_t20` were trained for 80 epochs; the other seven for 600 (640 for `sp20_t10`).

---

## Planned

- [ ] Recover or rewrite the Fig. 7 sweep-plot script from the nine `metrics.json` files
- [ ] Move the forward-script defaults to paths relative to the script
- [ ] Add a smoke test that loads each `model.pt` and reproduces its `metrics.json` medians
