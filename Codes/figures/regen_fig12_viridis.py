"""Regenerate Fig. 12(a-c) in the ORIGINAL viridis style used in v2_pinn_cs_july7_nm.docx.

Model: exp2 headline run sp20_t20_silu_F256_h256x256x256_a1_hpc (20% spatial / 20% temporal).
Ground truth from 3D_Pristine.mat (all 180,000 points = 3 plies).

This is a reconstruction of the published style, not a re-run of the original code -- the script
that produced the embedded viridis figures is not in this workspace. Everything below was
measured off the embedded rasters (word/media/image14-16.png):

  * SI units throughout: displacements in m, axes in m (NOT mm). Confirmed against the data --
    the published w colourbar at t=10.01 us reads -1.0 to 3.0 x1e-7, and the actual combined
    actual/predicted range is -1.422e-07 .. 3.955e-07 m.
  * y axis is CENTRED on zero (-.10 .. .10) even though the plate's true coordinates are
    y = -0.1995 .. -0.0005 m. Reproduced deliberately so the figure is a drop-in replacement;
    Y_SHIFT below is what does it. Set Y_SHIFT = 0.0 for true plate coordinates.
  * viridis, scaled to the data range (not symmetric about zero). Actual and predicted share
    one range per column -- concatenate both, take min/max -- which is the same convention as
    Codes/forward/reconstruct_field.py. The error panel is scaled 0..max.
  * every panel carries its own colourbar; offset text (1e-7 etc.) sits above it.
  * grey panes with white gridlines and black axis lines; serif text.

    python regen_fig12_viridis.py                  # all three components
    python regen_fig12_viridis.py --components w   # just Fig. 12c
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
from matplotlib.ticker import MaxNLocator, ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Codes", "forward"))
import pics_pinn as P  # noqa: E402
import h5py  # noqa: E402

MAT = os.path.abspath(os.path.join(HERE, "..", "..", "Dataset", "3D_Pristine.mat"))
RUN = os.path.join(REPO, "Results", "forward_runs",
                   "sp20_t20_silu_F256_h256x256x256_a1_hpc")

N_ROWS = 180000                      # 3 plies x 60,000 points
X_MIN, X_MAX = -0.1495, 0.1495       # m
Y_MIN, Y_MAX = -0.1995, -0.0005      # m, raw plate coordinates
Y_SHIFT = 0.1                        # centres y on zero, as the published figure does
NX, NY = 300, 200                    # 1 mm grid
Z_LAYERS = [0.0, -0.001, -0.002]     # ply mid-planes, m

TIMESTEPS = [1999, 3999, 5999]       # t = (T+1)*dt -> 20.00, 40.00, 60.00 us
COMP_IDX = {"u": 0, "v": 1, "w": 2}
COMP_MAT = {"u": "Disp_x", "v": "Disp_y", "w": "Disp_z"}
TAG = {"u": "12a", "v": "12b", "w": "12c"}

CMAP = "viridis"                     # signed rows: actual / predicted
# viridis is sequential, so like jet it carries no "white = zero" implication and the asymmetric
# data-range limits are fine as they stand (a diverging map such as RdBu_r would not be). Same
# map on the error row, matching the published figure where all nine panels share one scheme.
CMAP_ERROR = "viridis"
PANE = "#EAEAEA"                     # grey pane
GRID = "white"                       # white gridlines on the panes
AXLINE = "black"

# ---- view / box shape -------------------------------------------------------------------------
# azim = -120 reproduces the published arrangement: z axis on the far left, "y direction" along
# the bottom-left edge running .10 -> -.10 left to right, "x direction" along the bottom-right
# running -.15 -> .15, and the plate's (x_min, y_min) corner nearest the viewer. mplot3d's -60
# default puts x and y on the opposite edges and z on the right.
# For reference, the published rasters were measured (the plies share a footprint and differ only
# by a z offset, so the stack's upper silhouette is the top ply's two upper edges, whose pixel
# slopes give sin(elev) twice over): Fig 12a 26.0/24.8 deg, Fig 12b 24.5/23.8 deg, i.e. ~25 deg,
# with Lx/Ly ~ 0.76. We deliberately do NOT use either of those numbers -- 25 deg reads too
# edge-on here and 0.76 misrepresents the plate as narrower in x than in y.
#
# 26 deg / -137 deg views the stack more from the side than from above, which is what keeps the
# second and third plies clear of the top one. Plies separate when the per-ply vertical offset
# beats the plate's projected depth, and that ratio goes as 1/tan(elev) -- so lowering elevation
# buys visibility, while compressing z spends it. 26 deg roughly pays for the z = 1.18 below.
ELEV, AZIM = 26.0, -137.0
AXIS_SWAP = False
INVERT_XY = False
# x:y unchanged at ~3:2, the plate's true 300 x 200 mm footprint. z = 1.6181 was solved for, not
# guessed: it puts the drawn ply-to-ply gap at 8.5 mm on the page at 300 dpi. The gap is not
# linear in z (a taller box is scaled down to fit the panel rect), so changing the target means
# re-solving rather than scaling. The plies are physically 1 mm apart, so this is a ~8.5x
# exploded view.
BOX_ASPECT = (3.2, 2.2, 1.6181)
ZOOM = 1.0

# ---- font sizes (points, at FIGSIZE below) ----------------------------------------------------
FIGSIZE = (13.4, 10.0)
FS_COLHEAD = 15
FS_ROWLBL = 16
FS_AXLBL = 9.5
FS_TICK = 7.0
FS_CBTICK = 7.0
ROWLBL_X = 0.012                     # row labels: aligned, near the first column, not crowding it

# ---- manual layout (figure fractions) ---------------------------------------------------------
# With BOX_ASPECT set explicitly, mplot3d preserves that aspect and fits it inside whatever rect
# it is given, so the rect's own shape no longer drives how the plies read -- it only sets how
# much of the panel the box fills. Rows are spaced so the bottom row's y label clears the figure
# edge; 3D content is not clipped to the axes rect, so it can overflow into the margin.
# Panels are deliberately a little smaller than the space allows: the extra column and row pitch
# is the white space that makes the grid read cleanly at journal size.
L = 0.068                            # left edge of the first 3D panel (clears the z axis
                                     # AND leaves the row labels room)
COL_PITCH = 0.2538                   # panel + 5.5 mm + bar + 25 mm to the next panel
PANEL_W = 0.158                      # -10% width
CB_GAP, CB_W = 0.0162, 0.0061        # 5.5 mm panel->bar, bar 2.1 mm wide
TOP = 0.925                          # lowered, so there is more air above the column titles
ROW_PITCH = 0.2362                   # 7.5 mm between row rectangles
PANEL_H = 0.2067                     # 52.5 mm

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})


def panel_rect(r, c):
    return [L + c * COL_PITCH, TOP - r * ROW_PITCH - PANEL_H, PANEL_W, PANEL_H]


def cbar_rect(r, c, shrink=0.47):
    rect = panel_rect(r, c)
    h = rect[3] * shrink
    return [rect[0] + PANEL_W + CB_GAP, rect[1] + (rect[3] - h) / 2.0, CB_W, h]


def to_grid(vals, x, y):
    """Scatter one ply's per-point values onto the regular 1 mm (NY, NX) plate grid."""
    g = np.full((NY, NX), np.nan)
    # y arrives already shifted by Y_SHIFT, so the origin here must be shifted too; using the
    # raw Y_MIN pushes iy to 100..299 and the iy < NY test below then drops half the plate.
    ix = np.rint((x - X_MIN) * 1000.0).astype(int)
    iy = np.rint((y - (Y_MIN + Y_SHIFT)) * 1000.0).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    if not ok.all():   # loud, because dropping points here just renders a partial plate
        raise ValueError(f"to_grid dropped {(~ok).sum()} of {ok.size} points: "
                         f"x {x.min():.4f}..{x.max():.4f}, y {y.min():.4f}..{y.max():.4f} "
                         f"does not fit the {NY} x {NX} grid anchored at "
                         f"({X_MIN:.4f}, {Y_MIN + Y_SHIFT:.4f})")
    g[iy[ok], ix[ok]] = vals[ok]
    return g


def draw_panel(ax, vals, gx, gy, gz, XG, YG, norm, cmap_name=None):
    cmap = plt.get_cmap(cmap_name or CMAP)
    A, B = (YG, XG) if AXIS_SWAP else (XG, YG)
    for zl in Z_LAYERS:
        m = gz == zl
        g = to_grid(vals[m], gx[m], gy[m])
        ax.plot_surface(A, B, np.full_like(XG, zl), facecolors=cmap(norm(g)),
                        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

    xlim = (X_MIN, X_MAX)
    ylim = (Y_MIN + Y_SHIFT, Y_MAX + Y_SHIFT)
    xl_ = ylim if AXIS_SWAP else xlim
    yl_ = xlim if AXIS_SWAP else ylim
    if INVERT_XY:
        xl_, yl_ = xl_[::-1], yl_[::-1]
    ax.set_proj_type("persp")
    ax.set_xlim(*xl_)
    ax.set_ylim(*yl_)
    ax.set_zlim(-0.0021, 0.0001)
    if BOX_ASPECT is not None:
        ax.set_box_aspect(BOX_ASPECT, zoom=ZOOM)
    elif ZOOM != 1.0:
        ax.set_box_aspect(None, zoom=ZOOM)
    ax.view_init(elev=ELEV, azim=AZIM)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE)
        axis.pane.set_alpha(1.0)
        axis.pane.set_edgecolor(PANE)
        axis.line.set_color(AXLINE)
        axis.line.set_linewidth(1.0)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.5
    ax.grid(True)

    xt, xl = [-0.15, 0.0, 0.15], ["$-$.15", "0", ".15"]
    yt, yl = [-0.10, 0.0, 0.10], ["$-$.10", "0", ".10"]
    if AXIS_SWAP:
        xt, xl, yt, yl = yt, yl, xt, xl
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_zticks(Z_LAYERS)
    # z is written as 0/-1/-2 with a separate "x10^-3" header, the way the published panels do it;
    # mplot3d's own offset text would stack on top of the rotated z label instead.
    ax.set_zticklabels(["0", "$-$1", "$-$2"])
    ax.tick_params(labelsize=FS_TICK, pad=0.5)
    ax.tick_params(axis="z", pad=1.0)
    ax.text2D(0.055, 0.86, "$\\times\\,10^{-3}$", transform=ax.transAxes,
              ha="left", va="bottom", fontsize=FS_TICK)

    xlbl, ylbl = ("y direction (m)", "x direction (m)") if AXIS_SWAP else \
                 ("x direction (m)", "y direction (m)")
    # x can sit tight against its axis; y needs a touch more clearance because the two near-corner
    # tick labels (-.10 and -.15) meet right where it would land
    ax.set_xlabel(xlbl, fontsize=FS_AXLBL, labelpad=-4)
    ax.set_ylabel(ylbl, fontsize=FS_AXLBL, labelpad=-4)
    # z sits between the previous column's colourbar (outward) and the y label (inward); the wider
    # COL_PITCH and thinner bar leave room to keep it clear of both
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("z direction (m)", fontsize=FS_AXLBL, labelpad=0, rotation=90)


def add_colorbar(fig, rect, norm, cmap_name=None):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name or CMAP), norm=norm)
    cb = fig.colorbar(sm, cax=cax)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    cb.ax.tick_params(labelsize=FS_CBTICK, length=2.2, width=0.6, pad=1.5)
    # the published bars are headed by a bare power of ten, not matplotlib's "x10^n"
    off = cb.ax.yaxis.get_offset_text()
    off.set_fontsize(FS_CBTICK)
    fig.canvas.draw()
    off.set_text(off.get_text().replace("$\\times$", "").replace("\\times", ""))
    cb.outline.set_linewidth(0.6)
    return cb


CACHE = os.path.join(HERE, "fig12_cache.npz")


def load_cached_or_compute(refresh=False):
    """Ground truth + prediction, (3 timesteps, N_ROWS, 3), in metres.

    Inference over 180k points plus the .mat reads costs ~100 s, so the result is cached; style
    edits then re-render in seconds. --refresh recomputes.
    """
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE)
        # a cache built for different timesteps must not be reused silently
        cached = list(z["timesteps"]) if "timesteps" in z.files else None
        if cached == list(TIMESTEPS):
            print(f"[fig12v] loaded cache {os.path.basename(CACHE)}", flush=True)
            return z["gt"], z["pred"], z["gx"], z["gy"], z["gz"], list(z["times_us"])
        print(f"[fig12v] cache is for timesteps {cached}, need {TIMESTEPS} -- recomputing",
              flush=True)

    with h5py.File(MAT, "r") as f:
        DT = float(np.array(f["dt"]).ravel()[0])
        gx = np.array(f["X_zero_coord_ply"]).ravel()[:N_ROWS]
        gy = np.array(f["Y_zero_coord_ply"]).ravel()[:N_ROWS]
        gz = np.array(f["Z_zero_coord_ply"]).ravel()[:N_ROWS]

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

    MM = 1000.0
    gxm, gym, gzm = gx * MM, gy * MM, gz * MM      # the model is trained in mm
    pred = np.empty((len(TIMESTEPS), N_ROWS, 3))
    for i, T in enumerate(TIMESTEPS):
        t0 = time.time()
        t_phys = (T + 1) * DT
        for s in range(0, N_ROWS, 16384):
            e = min(s + 16384, N_ROWS)
            X = sc.encode(gxm[s:e], gym[s:e], gzm[s:e], np.full(e - s, t_phys))
            Xg = torch.tensor(X, dtype=dtype).requires_grad_(True)
            pred[i, s:e] = sc.decode_fields(
                P.displacement(model, Xg, sc.beta_y, sc.beta_z).detach().cpu().numpy())
        print(f"[fig12v] predicted step {T} ({time.time() - t0:.1f}s)", flush=True)
    pred /= MM                                      # model outputs mm -> m

    gt = np.empty((len(TIMESTEPS), N_ROWS, 3))
    with h5py.File(MAT, "r") as f:
        for ci, comp in enumerate("uvw"):
            t0 = time.time()
            gt[:, :, ci] = f[COMP_MAT[comp]][:N_ROWS, TIMESTEPS].T
            print(f"[fig12v] read {COMP_MAT[comp]} ({time.time() - t0:.1f}s)", flush=True)

    times_us = [(T + 1) * DT * 1e6 for T in TIMESTEPS]
    np.savez_compressed(CACHE, gt=gt, pred=pred, gx=gx, gy=gy + Y_SHIFT, gz=gz,
                        times_us=np.array(times_us), timesteps=np.array(TIMESTEPS))
    return gt, pred, gx, gy + Y_SHIFT, gz, times_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", nargs="+", default=["u", "v", "w"], choices=list(COMP_IDX))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--refresh", action="store_true", help="recompute instead of using the cache")
    args = ap.parse_args()

    gt, pred, gx, gy, gz, times_us = load_cached_or_compute(refresh=args.refresh)

    xs = X_MIN + np.arange(NX) / 1000.0
    ys = (Y_MIN + Y_SHIFT) + np.arange(NY) / 1000.0
    XG, YG = np.meshgrid(xs, ys)

    meta = {"model_run": os.path.basename(RUN), "timesteps": TIMESTEPS, "times_us": times_us,
            "cmap": CMAP, "cmap_error": CMAP_ERROR, "units": "m",
            "y_shift_m": Y_SHIFT, "panels": {}}

    for comp in args.components:
        ci = COMP_IDX[comp]
        fig = plt.figure(figsize=FIGSIZE, facecolor="white")
        rel_l2 = []
        for c in range(len(TIMESTEPS)):
            a = gt[c, :, ci]
            p = pred[c, :, ci]
            err = np.abs(p - a)
            rel_l2.append(float(np.linalg.norm(p - a) / (np.linalg.norm(a) + 1e-30)))

            both = np.concatenate([a, p])
            norm = plt.Normalize(float(both.min()), float(both.max()))
            enorm = plt.Normalize(0.0, float(err.max()))

            for r, (vals, nrm, cmn) in enumerate(((a, norm, CMAP), (p, norm, CMAP),
                                                  (err, enorm, CMAP_ERROR))):
                ax = fig.add_axes(panel_rect(r, c), projection="3d")
                # transparent, else the next row's patch paints over this row's overflowing
                # x/y labels (3D content is not clipped to the axes rect, the patch just hides it)
                ax.set_facecolor("none")
                draw_panel(ax, vals, gx, gy, gz, XG, YG, nrm, cmap_name=cmn)
                if r == 0:
                    ax.text2D(0.5, 1.02, f"$t = {times_us[c]:.2f}\\mu s$", transform=ax.transAxes,
                              ha="center", va="bottom", fontsize=FS_COLHEAD)
                add_colorbar(fig, cbar_rect(r, c), nrm, cmap_name=cmn)

        for r, lbl in enumerate((f"Actual ${comp}$", f"Predicted ${comp}$", "Errors")):
            rect = panel_rect(r, 0)
            fig.text(ROWLBL_X, rect[1] + rect[3] / 2.0, lbl, rotation=90,
                     va="center", ha="center", fontsize=FS_ROWLBL)

        stem = os.path.join(HERE, f"fig{TAG[comp]}_{comp}_viridis")
        # bbox_inches="tight": mplot3d draws axis labels outside the axes rect, so the bottom
        # row's "y direction (m)" runs past the figure edge on a fixed-size canvas
        fig.savefig(f"{stem}.png", dpi=args.dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        meta["panels"][comp] = {"rel_l2": rel_l2, "file": os.path.basename(f"{stem}.png")}
        print(f"[fig12v] saved {stem}.png  rel-L2 = "
              f"{', '.join('%.4f' % v for v in rel_l2)}", flush=True)

    json.dump(meta, open(os.path.join(HERE, "fig12_viridis_meta.json"), "w"), indent=2)
    print("[fig12v] done", flush=True)


if __name__ == "__main__":
    main()
