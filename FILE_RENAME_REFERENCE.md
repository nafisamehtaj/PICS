# File Reference

Where every file in this repository came from. Filenames were kept; only locations changed.
Source roots on the development machine:

- `SP` = `signal_reconstruction/scinonet-pinn/experiments/exp2_layerwise`
- `MP` = `signal_reconstruction/Manuscript Preparation`

| Current location | Original location |
|---|---|
| `Codes/forward/*.py` | `SP/scripts/*.py` |
| `Codes/figures/*.py` | `MP/generated_elements_4july/*.py` |
| `Dataset/dataset_layerwise_*_meta.json` | `SP/data/` |
| `Results/forward_runs/<tag>/` | `SP/outputs/<tag>/` (nine `sp{1,10,20}_t{1,10,20}_silu_F256_h256x256x256_a1_hpc` runs) |
| `Results/figures/*` | `MP/generated_elements_4july/*.png,*.svg,*_meta.json` and `SP/outputs/*.png,*.svg` |
| `LICENSE` | MIT, same text as the i-MAPS SciNoNet release |

Not copied: `Dataset/3D_Pristine.mat` (place the 16 GB simulation file there yourself) and the
training CSVs (rebuild with `Codes/forward/gen_layerwise_dataset.py`).
