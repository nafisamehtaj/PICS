"""Regenerate Fig. 11(ii) (source->boundary u/v/w prediction) and its layout schematic.

Model: exp2 headline run sp20_t20_silu_F256_h256x256x256_a1_hpc (20% spatial / 20% temporal).
Ground truth is read straight from the 16 GB 3D_Pristine.mat (ply-0 surface, z=0).

Point placement: every panel must show a first-arrival wave packet, not just its leading edge.
The packet onset moves ~2.7 timesteps per mm of propagation distance and the burst lasts
~1800 steps, so how much of it survives inside the 6001-step record is set by x:

    (a) -19.5  (b) 36.5   unchanged
    (c) 93.5 -> 57.5      packet fully rings down (u burst 3739->5549, 7% of peak at edge)
    (d) 116.5 -> 80.5     onset, ramp, main lobe and one trough visible; tail clipped
                          (24% of peak still live at step 6000)

(d) trades a slightly clipped tail for reach, which is the point of the panel. The old
(c)=93.5 / (d)=116.5 showed only the leading edge of the arrival, which is what this fixes.
Prediction quality is unaffected across this range: relative-L2(u) = 0.032 at (c), 0.024 at (d).

Note the line no longer reaches the plate boundary (x=149.5), so the figure shows decay with
increasing propagation distance rather than decay all the way to the edge; the title reflects
that. Nothing in the record fully rings down before step 6000 (low-level reverberation
persists everywhere), so "complete packet" means the dominant first arrival, not zero tail.

    python regen_source_to_boundary.py
"""
import os, sys, json
import numpy as np, torch, matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(REPO, "Codes", "forward")
sys.path.insert(0, SCRIPTS)
import pics_pinn as P  # noqa: E402
import h5py

MAT = os.path.abspath(os.path.join(HERE, "..", "..", "Dataset", "3D_Pristine.mat"))
RUN = os.path.join(REPO, "Results", "forward_runs",
                   "sp20_t20_silu_F256_h256x256x256_a1_hpc")
MM = 1000.0; N_PLY = 60000
AC = P.AC

SRC = (-49.5, -98.5); Y_LINE = -48.5
XS = [-19.5, 36.5, 57.5, 80.5]                      # (c),(d) pulled in so each shows a packet
LETTERS = ["(a)", "(b)", "(c)", "(d)"]

# ---- coords + GT lookup from the .mat ----
with h5py.File(MAT, "r") as f:
    DT = float(np.array(f["dt"]).ravel()[0])
    gx = np.array(f["X_zero_coord_ply"]).ravel()[:N_PLY] * MM
    gy = np.array(f["Y_zero_coord_ply"]).ravel()[:N_PLY] * MM
KEY = {(round(float(a), 1), round(float(b), 1)): i for i, (a, b) in enumerate(zip(gx, gy))}

def gt_signal(px, py):
    r = KEY[(round(px, 1), round(py, 1))]
    with h5py.File(MAT, "r") as f:
        return np.stack([np.array(f["Disp_x"][r]) * MM,
                         np.array(f["Disp_y"][r]) * MM,
                         np.array(f["Disp_z"][r]) * MM], axis=1)

# ---- load trained model ----
cfg = json.load(open(os.path.join(RUN, "config.json")))
P.set_dtype(cfg["dtype"]); P.set_seed(cfg["seed"])
P.ACTIVATION = cfg["activation"]; P.NUM_FREQ = cfg["num_freq"]; P.HIDDEN_SIZES = list(cfg["hidden"])
P.BALANCE_ALPHA = cfg["balance_alpha"]; P.SUBSAMPLE_KEEP = cfg["temporal_keep"]
P.N_HELD_SPATIAL = cfg["n_held_spatial"]; P.BATCH_SIZE = cfg["batch_size"]
P.DATA_ONLY_EPOCHS = cfg["data_only_epochs"]; P.BDRY_ENABLE = bool(cfg["bdry"])
blob = torch.load(os.path.join(RUN, "model.pt"), map_location="cpu")
sc = P.PotentialScalers.__new__(P.PotentialScalers); sc.__dict__.update(blob["scalers"])
model = P.make_net(sc, cfg["seed"]); model.load_state_dict(blob["state_dict"]); model.eval()

def predict(px, py, z=0.0, nt=6001):
    t = np.arange(1, nt + 1) * DT
    X = sc.encode(np.full(nt, px), np.full(nt, py), np.full(nt, z), t)
    Xg = torch.tensor(X, dtype=next(model.parameters()).dtype).requires_grad_(True)
    return sc.decode_fields(P.displacement(model, Xg, sc.beta_y, sc.beta_z).detach().cpu().numpy())

pts = [(x, Y_LINE) for x in XS]
gts = [gt_signal(px, py) for px, py in pts]
prs = [predict(px, py) for px, py in pts]
peaks = [float(np.max(np.abs(g[:, 0]))) for g in gts]
for (px, py), g, p in zip(pts, gts, prs):
    print(f"{px:7.1f},{py}: peak_u={np.max(np.abs(g[:,0])):.3e} relL2_u={P.relative_l2(p[:,0],g[:,0]):.3f}")

comps = ["u", "v", "w"]
ylim = [1.05 * max(np.max(np.abs(np.concatenate([g[:, ci] for g in gts]))),
                   np.max(np.abs(np.concatenate([p[:, ci] for p in prs])))) for ci in range(3)]

# ---- signal grid ----
fig, axes = plt.subplots(len(pts), 3, figsize=(22, 3.6 * len(pts)), squeeze=False)
t = np.arange(gts[0].shape[0])
for i, (px, py) in enumerate(pts):
    for ci, comp in enumerate(comps):
        ax = axes[i][ci]
        ax.axhline(0, color="#B0B4B8", lw=0.8, zorder=0)
        ax.plot(t, gts[i][:, ci], color=AC["ink"], lw=2.0, label="Ground truth")
        ax.plot(t, prs[i][:, ci], color=AC["red"], lw=2.4, ls="--", alpha=0.9, label="PINN")
        ax.set_ylim(-ylim[ci], ylim[ci]); ax.set_xlim(t[0], t[-1])
        ax.text(0.025, 0.86, LETTERS[i], transform=ax.transAxes, fontsize=26, fontweight="bold", va="center")
        ax.set_ylabel(f"{comp} [mm]", fontsize=20)
        if ci == 0:
            ax.annotate(f"{LETTERS[i]} x={px:.1f}, y={py:.1f}", xy=(-0.34, 0.5),
                        xycoords="axes fraction", rotation=90, va="center", ha="center",
                        fontsize=19, color=AC["ink"])
        if i == 0:
            ax.set_title(f"component {comp}", fontsize=26)
        if i == 0 and ci == 2:
            ax.legend(frameon=False, loc="upper right", fontsize=18)
for ci in range(3):
    axes[-1][ci].set_xlabel("timestep index", fontsize=20)
fig.suptitle(f"Amplitude decay along y={Y_LINE:g} mm with increasing distance from the "
             f"source (x={SRC[0]:g})  (shared y-scale per component)", fontsize=26, y=0.995)
fig.tight_layout(rect=[0.02, 0, 1, 0.985])
fig.savefig(os.path.join(HERE, "source_to_boundary_uvw.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "source_to_boundary_uvw.svg"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- layout schematic (background points are an illustrative seeded sample) ----
rng = np.random.RandomState(7); bidx = rng.choice(N_PLY, 70, replace=False)
figL, axL = plt.subplots(figsize=(9, 6.2))
axL.add_patch(plt.Rectangle((-149.5, -199.5), 299.0, 199.0, fill=False, edgecolor="#4A4E52", lw=1.8))
axL.scatter(gx[bidx], gy[bidx], s=14, color="#3a3f45", alpha=0.85, zorder=2)
axL.plot([SRC[0], XS[0]], [SRC[1], Y_LINE], ls=":", color="#9AA0A6", lw=1.6, zorder=3)
axL.plot([XS[0], XS[-1]], [Y_LINE, Y_LINE], ls="--", color=AC["amber"], lw=3.0, zorder=3)
axL.scatter([SRC[0]], [SRC[1]], marker="*", s=520, color=AC["red"], edgecolor="white", lw=0.8, zorder=6, label="Source point")
axL.scatter(XS, [Y_LINE] * len(XS), s=230, color="#1f6f8b", edgecolor="white", lw=1.0, zorder=6, label="Prediction point")
for i, (x, let) in enumerate(zip(XS, LETTERS)):   # stagger: (b),(c),(d) sit close together
    axL.annotate(let, (x, Y_LINE), textcoords="offset points",
                 xytext=(0, 16 if i % 2 == 0 else -30), ha="center", fontsize=17)
axL.set_xlim(-160, 160); axL.set_ylim(-205, 8)
axL.set_xlabel("x [mm]", fontsize=18); axL.set_ylabel("y [mm]", fontsize=18)
axL.set_title(f"Prediction points on the line y = {Y_LINE:g} mm", fontsize=20)
axL.tick_params(axis="both", labelsize=14); axL.legend(frameon=True, fontsize=14, loc="upper right")
figL.tight_layout()
figL.savefig(os.path.join(HERE, "source_to_boundary_layout.png"), dpi=300, bbox_inches="tight", facecolor="white")
figL.savefig(os.path.join(HERE, "source_to_boundary_layout.svg"), bbox_inches="tight", facecolor="white")
plt.close(figL)

json.dump({"source": list(SRC), "y_line": Y_LINE,
           "points": [[float(x), float(Y_LINE)] for x in XS], "peak_u_mm": peaks,
           "model_run": os.path.basename(RUN),
           "note": "(c) 93.5->57.5 and (d) 116.5->80.5 so every panel shows a first-arrival "
                   "wave packet rather than only its leading edge. (c) fully rings down; "
                   "(d) has a slightly clipped tail (24% of peak at step 6000), traded for "
                   "reach. Line no longer extends to the plate boundary at x=149.5."},
          open(os.path.join(HERE, "source_to_boundary_meta.json"), "w"), indent=2)
print("done")
