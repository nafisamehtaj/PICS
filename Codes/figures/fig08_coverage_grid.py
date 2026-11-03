"""Regenerate the 3x3 spatial-vs-temporal sweep grid (predicted u, top ply, t = 58.0 us).

Reproduces Results/figures/predicted_u_3x3_grid.png. The script that made
that PNG is not in this workspace, so this is a reconstruction; it is validated by checking the
per-panel relative-L2 against the values printed on the published figure:

    spatial 1%   1.11  1.53  1.42
    spatial 10%  0.83  0.75  0.68
    spatial 20%  0.29  0.17  0.18

Rows = spatial keep ratio, columns = temporal keep ratio, one model per cell (nine runs, all
silu_F256_h256x256x256_a1_hpc). Displacements in mm, single shared colour scale and one colourbar.

    python fig08_coverage_grid.py                # jet
    python fig08_coverage_grid.py --cmap viridis
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Codes"))
import pics_pinn as P  # noqa: E402
import h5py  # noqa: E402

MAT = os.path.abspath(os.path.join(HERE, "..", "..", "Dataset", "3D_Pristine.mat"))
OUT = os.path.join(REPO, "Results", "forward_runs")

N_PLY = 60000                        # top ply only (z = 0)
X_MIN, X_MAX = -149.5, 149.5         # mm
Y_MIN, Y_MAX = -199.5, -0.5          # mm
NX, NY = 300, 200
MM = 1000.0

TIMESTEP = 5799                      # t = (T+1)*dt = 58.00 us
SPATIAL = [1, 10, 20]                # rows
TEMPORAL = [1, 10, 20]               # columns
RUN_FMT = "sp{s}_t{t}_silu_F256_h256x256x256_a1_hpc"

CACHE = os.path.join(HERE, "pred_u_3x3_cache.npz")


def load_model(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    P.set_dtype(cfg["dtype"])
    P.set_seed(cfg["seed"])
    P.ACTIVATION = cfg["activation"]
    P.NUM_FREQ = cfg["num_freq"]
    P.HIDDEN_SIZES = list(cfg["hidden"])
    P.BALANCE_ALPHA = cfg["balance_alpha"]
    P.BDRY_ENABLE = bool(cfg["bdry"])
    blob = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    sc = P.PotentialScalers.__new__(P.PotentialScalers)
    sc.__dict__.update(blob["scalers"])
    model = P.make_net(sc, cfg["seed"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, sc


def predict_u(model, sc, gx, gy, gz, t_phys):
    out = np.empty(len(gx))
    dtype = next(model.parameters()).dtype
    for s in range(0, len(gx), 16384):
        e = min(s + 16384, len(gx))
        X = sc.encode(gx[s:e], gy[s:e], gz[s:e], np.full(e - s, t_phys))
        Xg = torch.tensor(X, dtype=dtype).requires_grad_(True)
        out[s:e] = sc.decode_fields(
            P.displacement(model, Xg, sc.beta_y, sc.beta_z).detach().cpu().numpy())[:, 0]
    return out


def to_grid(vals, x, y):
    g = np.full((NY, NX), np.nan)
    ix = np.rint(x - X_MIN).astype(int)
    iy = np.rint(y - Y_MIN).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    if not ok.all():
        raise ValueError(f"to_grid dropped {(~ok).sum()} of {ok.size} points")
    g[iy[ok], ix[ok]] = vals[ok]
    return g


def load_cached_or_compute(refresh=False):
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE)
        if int(z["timestep"]) == TIMESTEP:
            print(f"[grid] loaded cache {os.path.basename(CACHE)}", flush=True)
            return z["pred"], z["gt"], z["gx"], z["gy"], float(z["t_us"])
        print(f"[grid] cache is for timestep {int(z['timestep'])}, need {TIMESTEP} -- recomputing",
              flush=True)

    with h5py.File(MAT, "r") as f:
        DT = float(np.array(f["dt"]).ravel()[0])
        gx = np.array(f["X_zero_coord_ply"]).ravel()[:N_PLY] * MM
        gy = np.array(f["Y_zero_coord_ply"]).ravel()[:N_PLY] * MM
        gz = np.array(f["Z_zero_coord_ply"]).ravel()[:N_PLY] * MM
        gt = np.array(f["Disp_x"][:N_PLY, TIMESTEP]) * MM
    t_phys = (TIMESTEP + 1) * DT
    print(f"[grid] t = {t_phys*1e6:.2f} us, top ply z range {gz.min()}..{gz.max()} mm", flush=True)

    pred = np.empty((len(SPATIAL), len(TEMPORAL), N_PLY))
    for i, sp in enumerate(SPATIAL):
        for j, tp in enumerate(TEMPORAL):
            run = os.path.join(OUT, RUN_FMT.format(s=sp, t=tp))
            t0 = time.time()
            model, sc = load_model(run)
            pred[i, j] = predict_u(model, sc, gx, gy, gz, t_phys)
            rel = np.linalg.norm(pred[i, j] - gt) / np.linalg.norm(gt)
            print(f"[grid] sp{sp}/t{tp}: rel-L2 = {rel:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    np.savez_compressed(CACHE, pred=pred, gt=gt, gx=gx, gy=gy,
                        timestep=TIMESTEP, t_us=t_phys * 1e6)
    return pred, gt, gx, gy, t_phys * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    pred, gt, gx, gy, t_us = load_cached_or_compute(refresh=args.refresh)

    # one shared symmetric scale across all nine panels, as the published figure uses
    M = float(np.percentile(np.abs(pred), 99.9))
    norm = plt.Normalize(-M, M)
    extent = [X_MIN - 0.5, X_MAX + 0.5, Y_MIN - 0.5, Y_MAX + 0.5]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 3, figsize=(14.6, 10.8), squeeze=False)
    im = None
    rel = np.zeros((3, 3))
    for i, sp in enumerate(SPATIAL):
        for j, tp in enumerate(TEMPORAL):
            ax = axes[i][j]
            g = to_grid(pred[i, j], gx, gy)
            im = ax.imshow(g, origin="lower", extent=extent, cmap=args.cmap, norm=norm,
                           aspect="equal", interpolation="nearest")
            rel[i, j] = np.linalg.norm(pred[i, j] - gt) / np.linalg.norm(gt)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{sp}% / {tp}%", fontsize=17, pad=6)
            ax.text(0.975, 0.03, f"rel-L2 = {rel[i,j]:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="#EEEEEE",
                              edgecolor="none", alpha=0.85))
            if j == 0:
                ax.set_ylabel(f"spatial {sp}%", fontsize=17, labelpad=10)

    for j, tp in enumerate(TEMPORAL):
        pos = axes[0][j].get_position()
        fig.text(pos.x0 + pos.width / 2.0, 0.935, f"temporal {tp}%", ha="center", va="bottom",
                 fontsize=17, fontweight="bold", color="#1F3D99")

    fig.suptitle(f"Predicted in-plane displacement $u$ — top ply ($z$ = 0), "
                 f"t = {t_us:.1f} $\\mu$s", fontsize=21, y=0.985)

    fig.subplots_adjust(left=0.055, right=0.885, top=0.905, bottom=0.02, wspace=0.06, hspace=0.20)
    cax = fig.add_axes([0.905, 0.10, 0.016, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("u  [mm]", fontsize=15)
    cb.ax.tick_params(labelsize=13)

    stem = os.path.join(HERE, f"predicted_u_3x3_grid_{args.cmap}")
    fig.savefig(f"{stem}.png", dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    json.dump({"timestep": TIMESTEP, "t_us": t_us, "cmap": args.cmap,
               "spatial": SPATIAL, "temporal": TEMPORAL, "rel_l2": rel.tolist(),
               "clip": M, "units": "mm"},
              open(os.path.join(HERE, "pred_u_3x3_meta.json"), "w"), indent=2)
    print(f"[grid] saved {stem}.png", flush=True)
    print("[grid] rel-L2 matrix (rows = spatial, cols = temporal):", flush=True)
    for i, sp in enumerate(SPATIAL):
        print(f"        spatial {sp:>2}%  " + "  ".join(f"{v:.2f}" for v in rel[i]), flush=True)


if __name__ == "__main__":
    main()
