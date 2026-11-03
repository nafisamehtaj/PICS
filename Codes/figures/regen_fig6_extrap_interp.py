# -*- coding: utf-8 -*-
"""Regenerate manuscript Fig. 6 -- extrap_interp_sp10_temporal.png / .svg

3x2 panel: reconstructed in-plane displacement u (ground truth vs PINN) at
10% spatial sampling for temporal sampling 1%, 10%, 20%.
  Left  column: EXTRAPOLATION, an unseen (spatially held-out) point.
  Right column: INTERPOLATION, a seen point, error scored on held-out timesteps.

Selected points (printed at run time, also written to extrap_interp_meta.json):
  seen   (interpolation): x = -77.5 mm, y = -113.5 mm, z = 0 (top ply), base_row 14513
  unseen (extrapolation): x =  28.5 mm, y = -102.5 mm, z = 0 (top ply), base_row 35702

No dataset rebuild: scalers are read from each model.pt; the point selection and
temporal keep-mask are reproduced from coordinates + seed (matching
gen_layerwise_dataset.py and scinonet_pinn.build_dataset).

Original driver ran from the session scratchpad as seen_figure.py; this file is
the same code recovered verbatim from the 2026-07-05 session transcript.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "forward"))
OUTBASE = os.path.abspath(os.path.join(HERE, "..", "..", "Results", "forward_runs"))
MAT     = os.path.abspath(os.path.join(HERE, "..", "..", "Dataset", "3D_Pristine.mat"))
OUTDIR  = HERE
sys.path.insert(0, SCRIPTS)
import scinonet_pinn as P
import h5py

# ---- architecture globals to match the trained checkpoints ----
P.set_dtype("float64")
P.ACTIVATION = "silu"; P.NUM_FREQ = 256; P.HIDDEN_SIZES = [256, 256, 256]; P.CONCAT_RAW = True
DT = P.DT; N_T = P.N_T; MM = 1000.0
N_PLY = 60000
X_MIN, X_MAX, Y_MIN, Y_MAX = -149.5, 149.5, -199.5, -0.5
SEED = 42; PCT = 10.0; BOUND_FRAC = 0.20; EDGE_TOL = 0.6

# ---------------- reproduce the 10% selection (gen_layerwise_dataset.py) --------
with h5py.File(MAT, "r") as f:
    x_all = np.array(f["X_zero_coord_ply"]).ravel() * MM
    y_all = np.array(f["Y_zero_coord_ply"]).ravel() * MM
    z_all = np.array(f["Z_zero_coord_ply"]).ravel() * MM
xf, yf = x_all[:N_PLY], y_all[:N_PLY]
budget = int(round(PCT / 100.0 * N_PLY))
on_edge = ((np.abs(xf - X_MIN) <= EDGE_TOL) | (np.abs(xf - X_MAX) <= EDGE_TOL)
           | (np.abs(yf - Y_MIN) <= EDGE_TOL) | (np.abs(yf - Y_MAX) <= EDGE_TOL))
edge_idx = np.where(on_edge)[0]; interior_idx = np.where(~on_edge)[0]
rng = np.random.RandomState(SEED)
n_bound = min(len(edge_idx), int(round(BOUND_FRAC * budget)))
bound_pick = rng.choice(edge_idx, n_bound, replace=False)
n_int = max(0, budget - n_bound)
int_pick = rng.choice(interior_idx, min(n_int, len(interior_idx)), replace=False)
base = np.sort(np.concatenate([bound_pick, int_pick]))          # base (x,y) indices in [0,60000)
is_bound_base = np.isin(base, edge_idx)
print(f"[sel] budget={budget} base={len(base)} ({n_bound} boundary + {len(int_pick)} interior)")

# all 3 plies, sorted spatial-point order = sort by (x,y,z) as load_full_signal does
sel = np.array(sorted(set(base.tolist()) | {i + N_PLY for i in base} | {i + 2 * N_PLY for i in base}))
xs, ys, zs = x_all[sel], y_all[sel], z_all[sel]
order = np.lexsort((zs, ys, xs))                                # sort by x, then y, then z
sel_sorted = sel[order]
xyz_sorted = np.stack([xs[order], ys[order], zs[order]], axis=1)
n_spatial = len(sel_sorted)
print(f"[sel] spatial points (3 plies) = {n_spatial}; rows = {n_spatial*N_T:,}")

# ---------------- xy_points (unique x,y, sorted by x,y) + spatial holdout --------
# mirror build_dataset: drop_duplicates on (x,y) over the sorted df, ordered by (x,y)
xy_all = np.stack([x_all[base], y_all[base]], axis=1)
oxy = np.lexsort((xy_all[:, 1], xy_all[:, 0]))
xy_points = xy_all[oxy]
is_boundary_point = is_bound_base[oxy]
base_sorted_by_xy = base[oxy]                                   # base row index per xy_points row
nbhd_of_point = np.zeros(len(xy_points), dtype=int)
exclude = set(np.where(is_boundary_point)[0].tolist())
held_idx = P._choose_spatial_holdout(xy_points, nbhd_of_point, P.N_HELD_SPATIAL if False else 10,
                                     SEED, exclude=exclude)
held_set = set(held_idx)
seen_xy_idx = [i for i in range(len(xy_points)) if i not in held_set and not is_boundary_point[i]]
print(f"[sel] held={len(held_idx)} seen(interior)={len(seen_xy_idx)}")

# candidate seen points via the repo's spread-pick, then choose the cleanest one
cand = P._spread_pick(seen_xy_idx, 20, nbhd_of_point, SEED)
# read GT (z=0 ply) for candidates, pick interior point with strongest u signal
def read_gt(base_row):
    with h5py.File(MAT, "r") as f:
        u = np.array(f["Disp_x"][base_row, :]) * MM
        v = np.array(f["Disp_y"][base_row, :]) * MM
        w = np.array(f["Disp_z"][base_row, :]) * MM
    return u, v, w
best = None
for i in cand:
    br = int(base_sorted_by_xy[i])       # ply-0 row (base index < 60000)
    px, py = xy_points[i]
    if abs(px) > 120 or py < -180 or py > -20:   # keep well inside the plate
        continue
    u, _, _ = read_gt(br)
    energy = float(np.ptp(u))
    if best is None or energy > best[0]:
        best = (energy, i, br, px, py)
_, sel_i, base_row, PX, PY = best
u_gt, v_gt, w_gt = read_gt(base_row)
GT = {"u": u_gt, "v": v_gt, "w": w_gt}
print(f"[pick] seen point xy_index={sel_i} base_row={base_row} (x={PX:.1f}, y={PY:.1f})")

# position of this (x,y,z=0) point in the (x,y,z)-sorted spatial order (for keep mask)
p_pos = int(np.where(sel_sorted == base_row)[0][0])
print(f"[pick] sorted spatial position p={p_pos}")

# ---------------- reproduce is_train (temporal keep) per point ------------------
# keep = RandomState(seed+1).rand(len(df)) < subsample_keep, df sorted by (x,y,z,t)
keep_rng = np.random.RandomState(SEED + 1)
rand_all = keep_rng.rand(n_spatial * N_T)                       # 108M floats (~0.8 GB)
rand_point = rand_all[p_pos * N_T:(p_pos + 1) * N_T]            # this point's per-timestep draws
del rand_all

# ---------------- per temporal level: load model, predict, plot -----------------
LEVELS = [("1", 0.01, "sp10_t1_silu_F256_h256x256x256_a1_hpc"),
          ("10", 0.10, "sp10_t10_silu_F256_h256x256x256_a1_hpc"),
          ("20", 0.20, "sp10_t20_silu_F256_h256x256x256_a1_hpc")]

def build_scalers(d):
    sc = P.PotentialScalers.__new__(P.PotentialScalers)
    sc.__dict__.update(d)
    return sc

def predict_u(ckpt_dir, px, py):
    ck = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu", weights_only=False)
    sc = build_scalers(ck["scalers"])
    net = P.make_net(sc, SEED)
    net.load_state_dict(ck["state_dict"]); net.eval()
    t_full = np.arange(1, N_T + 1) * DT
    X = sc.encode(np.full(N_T, px), np.full(N_T, py), np.full(N_T, P.SURFACE_Z), t_full)
    Xg = torch.tensor(X, dtype=torch.float64).requires_grad_(True)
    uvw_hat = P.displacement(net, Xg, sc.beta_y, sc.beta_z).detach().cpu().numpy()
    return (uvw_hat * sc.s_f)[:, 0]        # u component

# ---- UNSEEN (held-out) point: nearest held point to the reference (x=28, y=-103) ----
best_u = None
for i in held_idx:
    px, py = xy_points[i]
    d = (px - 28.0) ** 2 + (py + 103.0) ** 2
    if best_u is None or d < best_u[0]:
        best_u = (d, i, int(base_sorted_by_xy[i]), float(px), float(py))
_, uns_i, uns_row, UX, UY = best_u
u_gt_uns = read_gt(uns_row)[0]
print(f"[pick] unseen(held) point xy_index={uns_i} base_row={uns_row} (x={UX:.1f}, y={UY:.1f})")

# ---- predictions per temporal level for both points ----
res_seen, res_unseen = [], []
for tlabel, keep, tag in LEVELS:
    d = os.path.join(OUTBASE, tag)
    # seen: interpolation error on held-out timesteps
    pu_s = predict_u(d, PX, PY)
    is_train = rand_point < keep
    test_idx = np.where(~is_train)[0]
    rel_s = P.relative_l2(pu_s[test_idx], GT["u"][test_idx])
    res_seen.append({"tlabel": tlabel, "u": pu_s, "rel": rel_s})
    # unseen: held point, error over the full signal
    pu_u = predict_u(d, UX, UY)
    rel_u = P.relative_l2(pu_u, u_gt_uns)
    res_unseen.append({"tlabel": tlabel, "u": pu_u, "rel": rel_u})
    print(f"[pred] temporal {tlabel}% : seen rel-L2={rel_s:.3f} | unseen rel-L2={rel_u:.3f}")

# ---------------- combined figure: 3x2 (unseen | seen) --------------------------
import matplotlib as mpl
mpl.rcdefaults()
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 1.0, "figure.dpi": 150,
})
INK, RED = "#212529", "#DC2626"
t = np.arange(N_T)
fig, axes = plt.subplots(3, 2, figsize=(13.6, 7.0), sharex=True,
                         gridspec_kw={"hspace": 0.5, "wspace": 0.18})
fig.subplots_adjust(top=0.90)

def panel(ax, gt, pred, tlabel, rel, predlabel):
    ax.plot(t, gt, color=INK, lw=1.3, label="Ground truth")
    ax.plot(t, pred, color=RED, lw=1.6, ls="--", label=predlabel)
    ax.set_title(f"spatial 10%  /  temporal {tlabel}%", pad=6)
    ax.set_ylabel("u [mm]")
    ax.margins(x=0.01)
    ax.text(0.015, 0.08, f"rel-L2 = {rel:.2f}", transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))

for row in range(3):
    panel(axes[row][0], u_gt_uns, res_unseen[row]["u"], res_unseen[row]["tlabel"],
          res_unseen[row]["rel"], "PINN (unseen)")
    panel(axes[row][1], GT["u"], res_seen[row]["u"], res_seen[row]["tlabel"],
          res_seen[row]["rel"], "PINN (seen)")
axes[1][0].legend(frameon=False, loc="upper left")
axes[1][1].legend(frameon=False, loc="upper left")
axes[2][0].set_xlabel("timestep index")
axes[2][1].set_xlabel("timestep index")

fig.canvas.draw()
for col, txt in [(0, "Extrapolation (Unseen Spatial Points)"),
                 (1, "Interpolation (Seen Spatial Points)")]:
    bb = axes[0][col].get_position()
    fig.text((bb.x0 + bb.x1) / 2, 0.955, txt, ha="center", va="bottom",
             fontsize=16, fontweight="bold")

stem = os.path.join(OUTDIR, "extrap_interp_sp10_temporal")
fig.savefig(stem + ".png", dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(stem + ".svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
from PIL import Image
print(f"SAVED: {stem}.png  size={Image.open(stem + '.png').size}")

json.dump({"seen_point": [float(PX), float(PY)], "unseen_point": [float(UX), float(UY)],
           "seen_rel_u": {r["tlabel"]: r["rel"] for r in res_seen},
           "unseen_rel_u": {r["tlabel"]: r["rel"] for r in res_unseen}},
          open(os.path.join(OUTDIR, "extrap_interp_meta.json"), "w"), indent=2)
