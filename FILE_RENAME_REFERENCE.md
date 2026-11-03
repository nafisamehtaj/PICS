# File Reference

Where every file in this repository came from, and what it was called there.
Source roots on the development machine:

- `SP` = `signal_reconstruction/scinonet-pinn/experiments/exp2_layerwise`
- `MP` = `signal_reconstruction/Manuscript Preparation`

| Current name | Original name |
|---|---|
| `Codes/pics_pinn.py` | `SP/scripts/scinonet_pinn.py` |
| `Codes/step1_generate_dataset.py` | `SP/scripts/gen_layerwise_dataset.py` |
| `Codes/step2_train.py` | `SP/scripts/run_exp.py` |
| `Codes/step3_evaluate_checkpoint.py` | `SP/scripts/eval_from_ckpt.py` |
| `Codes/step4_reconstruct_field.py` | `SP/scripts/reconstruct_field.py` |
| `Codes/step5_results_table.py` | `SP/scripts/make_results_table.py` |
| `Codes/tools/plot_plate.py` | `SP/scripts/plot_plate.py` |
| `Codes/tools/make_run_cards.py` | `SP/scripts/make_run_cards.py` |
| `Codes/tools/run_search.py` | `SP/scripts/run_search.py` |
| `Codes/tools/run_phys_sweep.py` | `SP/scripts/run_phys_sweep.py` |
| `Codes/figures/fig06_extrap_interp.py` | `MP/generated_elements_4july/regen_fig6_extrap_interp.py` |
| `Codes/figures/fig08_coverage_grid.py` | `MP/generated_elements_4july/regen_pred_u_3x3_grid.py` |
| `Codes/figures/fig11_source_to_boundary.py` | `MP/generated_elements_4july/regen_source_to_boundary.py` |
| `Codes/figures/fig12_layers.py` | `MP/generated_elements_4july/regen_fig12_layers.py` |
| `Codes/figures/fig12_viridis.py` | `MP/generated_elements_4july/regen_fig12_viridis.py` |
| `configs/training_config.yaml` | written for this release from `SP/HYPERPARAMETERS.md` and the run `config.json` files |
| `configs/train_all_nine.sh` | written for this release from the run `config.json` files and `SP/run20.sbatch` |
| `Dataset/dataset_layerwise_*_meta.json` | `SP/data/` |
| `Results/forward_runs/<run>/` | `SP/outputs/<run>/`, the nine `sp{1,10,20}_t{1,10,20}_silu_F256_h256x256x256_a1_hpc` runs |
| `Results/figures/*` | `MP/generated_elements_4july/*.png, *.svg, *_meta.json` and `SP/outputs/*.png, *.svg` |
| `LICENSE` | MIT, same text as the i-MAPS SciNoNet release |

Only paths and imports were edited in the moved scripts; the model, losses and figures are
unchanged. Not copied: `Dataset/3D_Pristine.mat` (place the 16 GB simulation file there
yourself) and the training CSVs (rebuild with `Codes/step1_generate_dataset.py`).
