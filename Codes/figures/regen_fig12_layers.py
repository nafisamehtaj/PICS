"""Regenerate Fig. 12(a-c): full-plate reconstruction across the three through-thickness layers.

Model: exp2 headline run sp20_t20_silu_F256_h256x256x256_a1_hpc (20% spatial / 20% temporal).
Ground truth is read straight from the 16 GB 3D_Pristine.mat (all 180,000 points = 3 plies).

One figure per displacement component (12a = u, 12b = v, 12c = w). Each figure is a 3x3 grid:
rows = actual / PINN prediction / absolute error, columns = t = 10.01, 22.22, 55.89 us.
Every panel stacks the three plies (z = 0, -1, -2 mm) in an orthographic 3D box.

Colour scheme. The signed actual/PINN rows use the diverging red-blue map (RdBu_r) centred on
zero. The absolute-error row keeps |error| >= 0 but uses the sequential white-to-red map from the
same family, so the two share a visual language instead of clashing with viridis; zero error
renders white, matching the quiescent plate in the rows above.

Colour limits are symmetric about zero, +/-M, shared by the actual and PINN panels of a column so
the two are directly comparable, and shown by one colourbar spanning those two rows. M is the
CLIP_PCT percentile of |field| rather than the maximum: the displacement peaks sharply at the
source, so a full-range limit renders the wavefronts almost white under a diverging map. Pass
--clip-pct 100 for the untruncated range.

Layout is placed manually rather than through subplots, because matplotlib reserves a large
margin around 3D axes that cannot be recovered with tight_layout/subplots_adjust.

    python regen_fig12_layers.py                  # all three components
    python regen_fig12_layers.py --components u   # just Fig. 12a
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
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(REPO, "Codes", "forward")
sys.path.insert(0, SCRIPTS)
import scinonet_pinn as P  # noqa: E402
import h5py  # noqa: E402

MAT = os.path.abspath(os.path.join(HERE, "..", "..", "Dataset", "3D_Pristine.mat"))
RUN = os.path.join(REPO, "Results", "forward_runs",
                   "sp20_t20_silu_F256_h256x256x256_a1_hpc")

MM = 1000.0
N_ROWS = 180000                     # 3 plies x 60,000 points
X_MIN, X_MAX = -149.5, 149.5
Y_MIN, Y_MAX = -199.5, -0.5
NX, NY = 300, 200                   # 1 mm grid over the 300 x 200 mm plate
Z_LAYERS = [0.0, -1.0, -2.0]        # ply mid-planes, mm

TIMESTEPS = [1000, 2221, 5588]      # t = (T+1)*dt -> 10.01, 22.22, 55.89 us
COMP_IDX = {"u": 0, "v": 1, "w": 2}
COMP_MAT = {"u": "Disp_x", "v": "Disp_y", "w": "Disp_z"}
COMP_LBL = {"u": ("12a", "in-plane displacement $u$"),
            "v": ("12b", "in-plane displacement $v$"),
            "w": ("12c", "out-of-plane displacement $w$")}
ROW_LBL = ["Actual", "PINN prediction", "Absolute error"]

CMAP_SIGNED = "RdBu_r"
CMAP_ERROR = "Reds"

INK = "#1A1D21"
FRAME = "#9AA1A8"
BG = "#E9ECEF"                      # page tint, so a zero-displacement (white) ply stays visible
PLATE_EDGE = "#5E656B"              # thin outline giving each ply a definite extent

# Through-thickness exaggeration. The plate is 2 mm thick over 300 x 200 mm, so at true scale the
# three plies collapse onto one line; the z axis is stretched to separate them (an exploded view).
# Z_EXAG is that stretch as a box-aspect ratio against x = 3.0, y = 2.0.
#
# Plies stop occluding each other only when their projected vertical offset clears the projected
# depth of a plate, i.e. tan(ELEV) < Z_EXAG / (y_aspect * z_span) ~= Z_EXAG / 5.2. Lowering ELEV
# both satisfies that and flattens each ply, so it reads as a thin sheet rather than a slab;
# shrinking Z_EXAG alone does the opposite, collapsing the plies onto one another.
Z_EXAG = 1.45
ELEV = 13.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

# ---- manual figure layout (figure fractions) -------------------------------------------------
L, R = 0.058, 0.994                 # left margin (row labels) / right edge
TOP, BOT = 0.895, 0.045             # top of the first panel row / bottom of the last
PANEL_W = 0.263                     # 3D panel width
CB_GAP, CB_W = 0.021, 0.0085        # gap before the colourbar (clears the z ticks) / its width
COL_W = (R - L) / 3.0
ROW_H = (TOP - BOT) / 3.0
PANEL_H = ROW_H * 0.93             # < 1 so the 3D content of adjacent rows cannot collide
# matplotlib clips 3D content to the axes rectangle, so ZOOM must stay low enough that the
# projected plates and their tick labels fit inside it -- above ~1.35 the corners get cropped.
ZOOM = 1.32


def panel_rect(r, c):
    top = TOP - r * ROW_H
    return [L + c * COL_W, top - PANEL_H, PANEL_W, PANEL_H]


def cbar_rect(r0, r1, c, shrink=0.62):
    """Colourbar rectangle spanning panel rows r0..r1 of column c."""
    top = TOP - r0 * ROW_H
    bottom = TOP - r1 * ROW_H - PANEL_H
    h = (top - bottom) * shrink
    return [L + c * COL_W + PANEL_W + CB_GAP, (top + bottom) / 2.0 - h / 2.0, CB_W, h]


def to_grid(vals, x, y):
    """Scatter one ply's per-point values onto the regular 1 mm (NY, NX) plate grid."""
    g = np.full((NY, NX), np.nan)
    ix = np.rint(x - X_MIN).astype(int)
    iy = np.rint(y - Y_MIN).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    g[iy[ok], ix[ok]] = vals[ok]
    return g


RECT_X = [X_MIN - 0.5, X_MAX + 0.5, X_MAX + 0.5, X_MIN - 0.5, X_MIN - 0.5]
RECT_Y = [Y_MIN - 0.5, Y_MIN - 0.5, Y_MAX + 0.5, Y_MAX + 0.5, Y_MIN - 0.5]


def draw_panel(ax, vals, gx, gy, gz, XG, YG, cmap, norm, stride=1):
    """Draw the three plies as flat sheets, each with a thin outline, on bare x/y/z axes."""
    for zl in Z_LAYERS:
        m = gz == zl
        g = to_grid(vals[m], gx[m], gy[m])
        ax.plot_surface(XG, YG, np.full_like(XG, zl), facecolors=cmap(norm(g)),
                        rstride=stride, cstride=stride, shade=False, linewidth=0,
                        antialiased=False)
        # outline sits a hair above the sheet so it never z-fights with the surface
        ax.plot(RECT_X, RECT_Y, zs=zl + 0.015, color=PLATE_EDGE, lw=0.7, zorder=5)

    ax.set_proj_type("ortho")
    ax.set_zlim(-2.20, 0.20)
    ax.set_box_aspect((3.0, 2.0, Z_EXAG), zoom=ZOOM)
    ax.view_init(elev=ELEV, azim=-58)

    # keep the x/y/z axes and their ticks; drop only the box walls and the interior grid
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color(FRAME)
        axis.line.set_linewidth(0.8)
    ax.set_xticks([-150, 0, 150])
    ax.set_yticks([-200, -100, 0])
    ax.set_zticks(Z_LAYERS)
    ax.set_zticklabels(["0", "-1", "-2"])
    ax.tick_params(labelsize=7.5, colors=INK, length=2.5, width=0.6, pad=-2)
    ax.tick_params(axis="z", pad=0.5)
    ax.set_xlabel("$x$ [mm]", fontsize=9, labelpad=-7)
    ax.set_ylabel("$y$ [mm]", fontsize=9, labelpad=-6)
    ax.set_zlabel("$z$ [mm]", fontsize=9, labelpad=-9)


def add_colorbar(fig, rect, cmap, norm, label):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    cb = fig.colorbar(sm, cax=cax)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.tick_params(labelsize=8.5, length=2.2, width=0.6, pad=1.5)
    cb.ax.yaxis.get_offset_text().set_fontsize(8.5)
    cb.ax.yaxis.get_offset_text().set_position((3.1, 0))
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(FRAME)
    cb.set_label(label, fontsize=9.5, labelpad=4)
    return cb


def main():
    global Z_EXAG, ELEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", nargs="+", default=["u", "v", "w"], choices=list(COMP_IDX))
    ap.add_argument("--clip-pct", type=float, default=99.5,
                    help="percentile of |field| used as the symmetric colour limit (100 = full range)")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--no-rell2", action="store_true", help="omit the per-column relative-L2 label")
    ap.add_argument("--z-exag", type=float, default=Z_EXAG,
                    help="through-thickness stretch; lower = thinner-looking laminate")
    ap.add_argument("--elev", type=float, default=ELEV, help="3D view elevation, degrees")
    args = ap.parse_args()
    Z_EXAG, ELEV = args.z_exag, args.elev

    # ---- coordinates ----
    with h5py.File(MAT, "r") as f:
        DT = float(np.array(f["dt"]).ravel()[0])
        gx = np.array(f["X_zero_coord_ply"]).ravel()[:N_ROWS] * MM
        gy = np.array(f["Y_zero_coord_ply"]).ravel()[:N_ROWS] * MM
        gz = np.array(f["Z_zero_coord_ply"]).ravel()[:N_ROWS] * MM
    times_us = [(T + 1) * DT * 1e6 for T in TIMESTEPS]
    print(f"[fig12] dt={DT:g} s, timesteps={TIMESTEPS} -> t={['%.2f' % t for t in times_us]} us",
          flush=True)

    # ---- trained model ----
    cfg = json.load(open(os.path.join(RUN, "config.json")))
    P.set_dtype(cfg["dtype"])
    P.set_seed(cfg["seed"])
    P.ACTIVATION = cfg["activation"]
    P.NUM_FREQ = cfg["num_freq"]
    P.HIDDEN_SIZES = list(cfg["hidden"])
    P.BALANCE_ALPHA = cfg["balance_alpha"]
    P.BDRY_ENABLE = bool(cfg["bdry"])
    blob = torch.load(os.path.join(RUN, "model.pt"), map_location="cpu", weights_only=False)
    sc = P.PotentialScalers.__new__(P.PotentialScalers)
    sc.__dict__.update(blob["scalers"])
    model = P.make_net(sc, cfg["seed"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    dtype = next(model.parameters()).dtype

    # ---- predict every point at each timestep (u, v, w together) ----
    pred = {}
    for T in TIMESTEPS:
        t0 = time.time()
        out = np.empty((N_ROWS, 3))
        t_phys = (T + 1) * DT
        for s in range(0, N_ROWS, 16384):
            e = min(s + 16384, N_ROWS)
            X = sc.encode(gx[s:e], gy[s:e], gz[s:e], np.full(e - s, t_phys))
            Xg = torch.tensor(X, dtype=dtype).requires_grad_(True)
            out[s:e] = sc.decode_fields(
                P.displacement(model, Xg, sc.beta_y, sc.beta_z).detach().cpu().numpy())
        pred[T] = out
        print(f"[fig12] predicted step {T} ({time.time() - t0:.1f} s)", flush=True)

    xs = np.arange(NX) + X_MIN
    ys = np.arange(NY) + Y_MIN
    XG, YG = np.meshgrid(xs, ys)
    meta = {"model_run": os.path.basename(RUN), "timesteps": TIMESTEPS,
            "times_us": times_us, "clip_pct": args.clip_pct,
            "cmap_signed": CMAP_SIGNED, "cmap_error": CMAP_ERROR, "panels": {}}

    for comp in args.components:
        tag, pretty = COMP_LBL[comp]
        t0 = time.time()
        with h5py.File(MAT, "r") as f:
            gt = f[COMP_MAT[comp]][:N_ROWS, TIMESTEPS] * MM
        print(f"[fig12] read ground truth {comp} ({time.time() - t0:.1f} s)", flush=True)

        fig = plt.figure(figsize=(15.0, 7.6), facecolor=BG)
        rel_l2 = []
        for c, T in enumerate(TIMESTEPS):
            a = gt[:, c]
            p = pred[T][:, COMP_IDX[comp]]
            err = np.abs(p - a)
            rel = float(np.linalg.norm(p - a) / (np.linalg.norm(a) + 1e-30))
            rel_l2.append(rel)

            both = np.concatenate([a, p])
            M = max(float(np.percentile(np.abs(both), args.clip_pct)), 1e-30)
            norm = plt.Normalize(-M, M)
            # same percentile treatment on the error, whose maximum is likewise a source spike
            enorm = plt.Normalize(0.0, max(float(np.percentile(err, args.clip_pct)), 1e-30))

            for r, (vals, cmap, nrm) in enumerate(((a, CMAP_SIGNED, norm),
                                                   (p, CMAP_SIGNED, norm),
                                                   (err, CMAP_ERROR, enorm))):
                ax = fig.add_axes(panel_rect(r, c), projection="3d")
                ax.set_facecolor(BG)
                draw_panel(ax, vals, gx, gy, gz, XG, YG, plt.get_cmap(cmap), nrm)
                if r == 0:
                    ax.text2D(0.5, 1.01, f"t = {times_us[c]:.2f} $\\mu$s", transform=ax.transAxes,
                              ha="center", va="bottom", fontsize=15)
                if r == 2 and not args.no_rell2:
                    # top-left is the only reliably empty corner; bottom-left hits the x label
                    ax.text2D(0.02, 0.97, f"relative-L2 = {rel:.3f}", transform=ax.transAxes,
                              ha="left", va="top", fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                                        edgecolor=FRAME, linewidth=0.6))
                if c == 0:
                    fig.text(L - 0.014, panel_rect(r, 0)[1] + PANEL_H / 2.0, ROW_LBL[r],
                             rotation=90, va="center", ha="center", fontsize=14)

            add_colorbar(fig, cbar_rect(0, 1, c, shrink=0.42), CMAP_SIGNED, norm, f"${comp}$  [mm]")
            add_colorbar(fig, cbar_rect(2, 2, c, shrink=0.66), CMAP_ERROR, enorm, "$|$error$|$  [mm]")
            print(f"[fig12] {comp} t={times_us[c]:.2f} us: rel-L2 = {rel:.4f}", flush=True)

        fig.text(0.5, 0.972, f"Full-plate reconstruction of the {pretty} across the three "
                             f"through-thickness layers ($z$ = 0, $-$1, $-$2 mm)",
                 ha="center", va="center", fontsize=17)

        stem = os.path.join(HERE, f"fig{tag}_{comp}_layers")
        fig.savefig(f"{stem}.png", dpi=args.dpi, facecolor=BG)
        plt.close(fig)
        meta["panels"][comp] = {"rel_l2": rel_l2, "file": os.path.basename(f"{stem}.png")}
        print(f"[fig12] saved {stem}.png ({time.time() - t0:.1f} s)", flush=True)

    json.dump(meta, open(os.path.join(HERE, "fig12_layers_meta.json"), "w"), indent=2)
    print("[fig12] done", flush=True)


if __name__ == "__main__":
    main()
