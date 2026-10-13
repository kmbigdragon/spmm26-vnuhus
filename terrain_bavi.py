"""
Population spreading over the real terrain of Ba Vi (Hanoi).

Model (same as terrain2d.py, but the landscape comes from real data):

    dN/dt = div( D(x) grad N )                (movement, slowed by steep slopes and rivers)
          - div( chi(x) N grad A(x) )         (drift toward better habitat)
          + r(x) N (1 - N / K(x))             (local logistic growth)

with no-flux boundary conditions. Units: x, y in km, t in years.

Terrain functions, built from the elevation z(x, y) and the river network:
    slope   s(x, y) = |grad z|                         (m per m)
    D(x, y) = D0 * exp(-s / s0) * river factor         mobility
    K(x, y) = K0 * exp(-(z / z_c)^2)                   habitat gets poorer higher up
    r(x, y) = r0 on land, < 0 in wide rivers           crossing a big river is risky
    A(x, y) = K / K0,  chi = chi0 * D / D0             attraction to good habitat

Rivers (OpenStreetMap), from barrier to minor obstacle:
    Song Hong, Song Da : wide, D almost 0 and mortality -> they block the population
    Song Tich, others  : narrow, D reduced              -> they only slow it down

Needs the data written by real_terrain.py (run `python real_terrain.py bavi` first):
    data/dem_bavi.npz, data/rivers_bavi.json

Numerics: cell-centred finite volumes + explicit Euler (as in terrain2d.py).

Run:  python terrain_bavi.py              -> FigB1..FigB3 png files
      python terrain_bavi.py --animate    -> also shows a live animation
"""

import argparse
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource, LinearSegmentedColormap, TwoSlopeNorm

# ============================================================
# 1. PARAMETERS
# ============================================================

D0 = 1.0                       # mobility on flat land [km^2 / year]
r0 = 0.5                       # intrinsic growth rate [1 / year]
K0 = 100.0                     # carrying capacity in the lowland
chi0 = 1.0                     # strength of attraction toward good habitat [km^2 / year]

s0 = 0.2                       # slope scale: D = D0 exp(-slope / s0)
z_c = 700.0                    # altitude scale of habitat quality [m]

# rivers: (text in the OSM name, half-width in cells, D factor, r inside)
RIVER_CLASSES = [
    ("Sông Hồng", 2, 0.002, -2.0),     # about 700 m wide on the grid
    ("Sông Đà", 1, 0.002, -2.0),       # about 450 m wide
    ("Sông Tích", 0, 0.1, 0.0),        # one cell, slows the population down
    ("", 0, 0.3, 0.0),                 # any other river
]

START = (21.10, 105.40)        # (lat, lon) where the population starts
N_init, sigma_init = 80.0, 0.5     # Gaussian bump, sigma in km

T_end = 30.0                   # years
snapshot_times = [0, 3, 6, 10, 18, 30]
arrival_threshold = 0.2 * K0   # "the population has arrived" when N >= this

COARSEN = 5                    # 5 x 5 DEM pixels (about 30 m) -> one cell (about 150 m)
DEM_FILE = "data/dem_bavi.npz"
RIVER_FILE = "data/rivers_bavi.json"
R_EARTH = 6371.0               # km

# ============================================================
# 2. GRID FROM THE DEM
# ============================================================

dem = np.load(DEM_FILE)
b = COARSEN
nr = (dem["z"].shape[0] // b) * b
nc = (dem["z"].shape[1] // b) * b


def block_mean(a):
    return a.reshape(-1, b).mean(axis=1)


# average b x b pixels per cell; flip rows so that row 0 is the south edge
Z = dem["z"][:nr, :nc].reshape(nr // b, b, nc // b, b).mean(axis=(1, 3))[::-1]
lat = block_mean(dem["lat"][:nr])[::-1]          # south -> north
lon = block_mean(dem["lon"][:nc])                # west -> east
x = block_mean(dem["x_km"][:nc])                 # km east of the west edge
y = block_mean(dem["y_km"][:nr])[::-1]           # km north of the south edge
Ny, Nx = Z.shape
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)                         # arrays are indexed [j, i] = [y, x]
Lx, Ly = x[-1] + dx / 2, y[-1] + dy / 2

LAT0, LON0 = dem["lat"].min(), dem["lon"].min()
lat_c = math.radians(dem["lat"].mean())


def to_km(lat_deg, lon_deg):
    """Same projection as real_terrain.py."""
    xk = R_EARTH * math.cos(lat_c) * np.radians(np.asarray(lon_deg) - LON0)
    yk = R_EARTH * np.radians(np.asarray(lat_deg) - LAT0)
    return xk, yk


x_init, y_init = to_km(*START)

# ============================================================
# 3. RIVERS ON THE GRID
# ============================================================


def rasterize(points_lat, points_lon):
    """Mark every cell crossed by a polyline."""
    mask = np.zeros((Ny, Nx), dtype=bool)
    px, py = to_km(points_lat, points_lon)
    step = 0.25 * min(dx, dy)
    for k in range(len(px) - 1):
        n = max(2, int(math.hypot(px[k + 1] - px[k], py[k + 1] - py[k]) / step) + 1)
        sx = np.linspace(px[k], px[k + 1], n)
        sy = np.linspace(py[k], py[k + 1], n)
        i = np.round((sx - x[0]) / dx).astype(int)
        j = np.round((sy - y[0]) / dy).astype(int)
        ok = (i >= 0) & (i < Nx) & (j >= 0) & (j < Ny)
        mask[j[ok], i[ok]] = True
    return mask


def widen(mask, radius):
    """Grow a mask by `radius` cells in every direction (no wrap-around)."""
    padded = np.pad(mask, radius)
    out = mask.copy()
    for dj in range(-radius, radius + 1):
        for di in range(-radius, radius + 1):
            if di * di + dj * dj <= radius * radius:
                out |= padded[radius + dj:radius + dj + Ny, radius + di:radius + di + Nx]
    return out


def load_rivers():
    """One mask per river class (in the order of RIVER_CLASSES), and a label
    position (x, y in km) for every named river."""
    with open(RIVER_FILE, encoding="utf-8") as f:
        ways = json.load(f)["elements"]
    masks = [np.zeros((Ny, Nx), dtype=bool) for _ in RIVER_CLASSES]
    points = {}
    for way in ways:
        name = way.get("tags", {}).get("name", "")
        c = next(k for k, (key, *_) in enumerate(RIVER_CLASSES) if key in name)
        pts = way.get("geometry", [])
        masks[c] |= rasterize([p["lat"] for p in pts], [p["lon"] for p in pts])
        if RIVER_CLASSES[c][0]:
            px, py = to_km([p["lat"] for p in pts], [p["lon"] for p in pts])
            inside = (px > 1) & (px < Lx - 1) & (py > 1) & (py < Ly - 1)
            points.setdefault(RIVER_CLASSES[c][0], []).extend(zip(px[inside], py[inside]))
    # candidate label positions along each river (middle first), south to north
    labels = {}
    for name, p in points.items():
        p = sorted(p, key=lambda q: q[1])
        labels[name] = [p[int(q * (len(p) - 1))]
                        for q in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9)]
    return [widen(m, RIVER_CLASSES[c][1]) for c, m in enumerate(masks)], labels


# ============================================================
# 4. TERRAIN FUNCTIONS D, r, K, A, chi
# ============================================================

def build_landscape():
    gz_y, gz_x = np.gradient(Z, 1000 * dy, 1000 * dx)        # m per m
    slope = np.hypot(gz_x, gz_y)

    D = D0 * np.exp(-slope / s0)
    K = K0 * np.exp(-(Z / z_c) ** 2)
    r = np.full_like(Z, r0)

    # rivers: apply the minor classes first so the big rivers win where they meet
    river_masks, river_labels = load_rivers()
    river_any = np.zeros_like(Z, dtype=bool)
    big_river = np.zeros_like(Z, dtype=bool)
    for (key, _, D_factor, r_river), m in reversed(list(zip(RIVER_CLASSES, river_masks))):
        D[m] = D0 * D_factor
        r[m] = r_river
        river_any |= m
        if r_river < 0:
            big_river |= m

    A = K / K0
    chi = chi0 * D / D0            # animals drift as fast as they can move
    return dict(z=Z, slope=slope, D=D, r=r, K=K, A=A, chi=chi,
                river=river_any, big_river=big_river, river_labels=river_labels)


# ============================================================
# 5. SOLVER (finite volumes, same scheme as terrain2d.py)
# ============================================================

def harmonic_mean(a, b_):
    s = a + b_
    return np.where(s > 0, 2 * a * b_ / np.where(s > 0, s, 1.0), 0.0)


def face_coefficients(L):
    """Diffusivity and taxis velocity on the interior cell faces."""
    D, chi, A = L["D"], L["chi"], L["A"]
    Dfx = harmonic_mean(D[:, :-1], D[:, 1:])
    vfx = harmonic_mean(chi[:, :-1], chi[:, 1:]) * (A[:, 1:] - A[:, :-1]) / dx
    Dfy = harmonic_mean(D[:-1, :], D[1:, :])
    vfy = harmonic_mean(chi[:-1, :], chi[1:, :]) * (A[1:, :] - A[:-1, :]) / dy
    return Dfx, vfx, Dfy, vfy


def stable_dt(L, faces, safety=0.8):
    Dfx, vfx, Dfy, vfy = faces
    Dmax = max(Dfx.max(), Dfy.max())
    rate = 2 * Dmax / dx ** 2 + 2 * Dmax / dy ** 2 \
        + np.abs(vfx).max() / dx + np.abs(vfy).max() / dy + np.abs(L["r"]).max()
    return safety / rate


def rhs(N, L, faces):
    """dN/dt of the PDE."""
    Dfx, vfx, Dfy, vfy = faces
    # diffusive fluxes F = -D grad N, plus upwind taxis fluxes
    Fx = -Dfx * (N[:, 1:] - N[:, :-1]) / dx + vfx * np.where(vfx > 0, N[:, :-1], N[:, 1:])
    Fy = -Dfy * (N[1:, :] - N[:-1, :]) / dy + vfy * np.where(vfy > 0, N[:-1, :], N[1:, :])
    # no flux through the outer boundary
    Fx = np.pad(Fx, ((0, 0), (1, 1)))
    Fy = np.pad(Fy, ((1, 1), (0, 0)))
    divF = (Fx[:, 1:] - Fx[:, :-1]) / dx + (Fy[1:, :] - Fy[:-1, :]) / dy

    # logistic growth on land, pure mortality where r <= 0
    r = L["r"]
    growth = np.where(r > 0, r * N * (1.0 - N / L["K"]), r * N)
    return -divF + growth


def initial_condition():
    return N_init * np.exp(-((X - x_init) ** 2 + (Y - y_init) ** 2) / (2 * sigma_init ** 2))


def simulate(L, T=T_end, snaps=snapshot_times):
    faces = face_coefficients(L)
    dt = stable_dt(L, faces)
    Nt = int(np.ceil(T / dt))
    dt = T / Nt
    print(f"dt = {dt:.5f} years, steps = {Nt}")

    N = initial_condition()
    arrival = np.full(N.shape, np.nan)
    arrival[N >= arrival_threshold] = 0.0
    saved = {0: N.copy()}
    next_snaps = [s for s in snaps if s > 0]

    for n in range(1, Nt + 1):
        N = np.maximum(N + dt * rhs(N, L, faces), 0.0)
        t = n * dt
        newly = (N >= arrival_threshold) & np.isnan(arrival)
        arrival[newly] = t
        while next_snaps and t >= next_snaps[0] - 1e-9:
            saved[next_snaps.pop(0)] = N.copy()

    return dict(N=N, snaps=saved, arrival=arrival, dt=dt)


def front_speed(arrival, L, margin_km=1.5):
    """Local speed of the invasion front = 1 / |grad(arrival time)|  [km / year].

    Not shown where the front is not a travelling wave:
      * near the start (the front has not formed yet)
      * near the outer edge and the big rivers: when the front runs into a wall,
        the last strip fills up almost at once, which looks like a very fast front.
    """
    gT_y, gT_x = np.gradient(arrival, dy, dx)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = 1.0 / np.hypot(gT_x, gT_y)
    near_start = np.hypot(X - x_init, Y - y_init) < 2.0
    near_edge = (X < margin_km) | (X > Lx - margin_km) | (Y < margin_km) | (Y > Ly - margin_km)
    near_wall = widen(L["big_river"], int(round(margin_km / min(dx, dy))))
    speed[near_start | near_edge | near_wall] = np.nan
    return speed


# ============================================================
# 6. FIGURES
# ============================================================

EXTENT = [0, Lx, 0, Ly]
WATER = "#2B6CB0"
ELEV_CMAP = LinearSegmentedColormap.from_list(          # as in real_terrain.py
    "elevation", ["#f6f1e4", "#dcc9a0", "#b8925c", "#8a5f36", "#5a3920"])
# population: light green -> dark green (no white, so it stays visible on the relief)
POP_CMAP = LinearSegmentedColormap.from_list("population", plt.cm.Greens(np.linspace(0.3, 1, 256)))
SHOW_MIN = 5.0                 # cells with N below this are not drawn
# light greens for the attraction map, so that the drift arrows stand out
LIGHT_GREENS = LinearSegmentedColormap.from_list("light_greens", plt.cm.Greens(np.linspace(0, 0.55, 256)))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888",
                     "axes.labelcolor": "#222222", "xtick.color": "#555555",
                     "ytick.color": "#555555"})

hillshade = LightSource(azdeg=315, altdeg=45).hillshade(
    Z, vert_exag=2, dx=1000 * dx, dy=-1000 * dy)       # rows go south -> north


def draw_terrain(ax, L, shade=True):
    """Grey relief, 300 / 700 / 1100 m contours and the rivers."""
    if shade:
        ax.imshow(hillshade, extent=EXTENT, origin="lower", cmap="gray",
                  vmin=-0.3, vmax=1.3, alpha=0.9)
    ax.contour(X, Y, Z, levels=[300, 700, 1100], colors="#3b2a1e",
               linewidths=0.5, alpha=0.6)
    ax.contourf(X, Y, L["river"].astype(float), levels=[0.5, 1.5], colors=[WATER])
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


def mark_start(ax):
    ax.plot(x_init, y_init, marker="*", ms=10, color="white", mec="black", mew=0.9)


RIVER_NAMES = {"Sông Hồng": "Red River", "Sông Đà": "Da River", "Sông Tích": "Tich River"}


def label_features(ax, L, size=7):
    """Label Ba Vi and the named rivers without overlapping labels."""
    style = dict(fontsize=size, color="#111111",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    char_km, line_km = 1.0, 2.5          # rough size of one character / one line on the map

    def box(x_anchor, y_mid, text, ha):
        w = len(text) * char_km
        x0 = {"left": x_anchor, "right": x_anchor - w, "center": x_anchor - w / 2}[ha]
        return x0, x0 + w, y_mid - line_km / 2, y_mid + line_km / 2

    def free(b, placed):
        inside = b[0] > 0 and b[1] < Lx and b[2] > 0 and b[3] < Ly
        return inside and all(b[1] < p[0] or b[0] > p[1] or b[3] < p[2] or b[2] > p[3]
                              for p in placed)

    jp, ip = np.unravel_index(np.argmax(Z), Z.shape)
    xs, ys = X[jp, ip], Y[jp, ip] - 1.2 - line_km / 2
    ax.text(xs, ys, "Ba Vì", ha="center", va="center", **style)
    placed = [box(xs, ys, "Ba Vì", "center")]

    for osm_name, candidates in L["river_labels"].items():
        text = RIVER_NAMES.get(osm_name, osm_name)
        options = [(cx + 0.6, cy, "left") for cx, cy in candidates] \
            + [(cx - 0.6, cy, "right") for cx, cy in candidates]
        xa, ya, ha = next((o for o in options if free(box(o[0], o[1], text, o[2]), placed)),
                          options[0])
        placed.append(box(xa, ya, text, ha))
        ax.text(xa, ya, text, ha=ha, va="center", fontstyle="italic", **style)


FIG_W = 7.2                    # inches: figures are sized for a full slide width


def fig_pair(L, panels, fname, arrows=False, labels=False):
    """Two maps side by side; panels = [(field, cmap, (vmin, vmax), title, colorbar label)]."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.9), constrained_layout=True)
    for ax, (field, cmap, (vmin, vmax), title, cb_label) in zip(axes, panels):
        im = ax.imshow(field, extent=EXTENT, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        draw_terrain(ax, L, shade=False)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.9).set_label(cb_label)
    if labels:
        label_features(axes[0], L)
        mark_start(axes[0])
    if arrows:                                        # drift chi grad A
        gA_y, gA_x = np.gradient(L["A"], dy, dx)
        k = 10                                        # average over 10 x 10 cells (~1.5 km)
        ny, nx = Ny // k, Nx // k

        def blocks(a):
            return a[:ny * k, :nx * k].reshape(ny, k, nx, k).mean(axis=(1, 3))

        vx, vy = blocks(L["chi"] * gA_x), blocks(L["chi"] * gA_y)
        bx, by = blocks(X), blocks(Y)
        mag = np.hypot(vx, vy)
        keep = mag > 0.25 * mag.max()                 # only where the drift matters
        # black arrows with a white rim; the strongest drift is drawn 3.5 km long
        axes[1].quiver(bx[keep], by[keep], vx[keep], vy[keep],
                       angles="xy", scale_units="xy", scale=mag.max() / 3.5,
                       color="black", edgecolor="white", linewidth=0.5,
                       width=0.008, headwidth=3.5, headlength=4, zorder=4)
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    print("saved", fname)


def fig_landscape(L):
    fig_pair(L, [
        (L["z"], ELEV_CMAP, (0, Z.max()), "Elevation $z(x,y)$", "m"),
        (L["D"] / D0, "Greys_r", (0, 1), "Mobility $D/D_0 = e^{-s/s_0}$", "$D/D_0$"),
    ], "FigB1a_bavi_elevation_mobility.png", labels=True)
    fig_pair(L, [
        (L["K"], "Greens", (0, K0), "Capacity $K = K_0 e^{-(z/z_c)^2}$", "$K$"),
        (L["A"], LIGHT_GREENS, (0, 1), r"Attraction $A = K/K_0$, drift $\chi\nabla A$", "$A$"),
    ], "FigB1b_bavi_capacity_attraction.png", arrows=True)


def fig_snapshots(L, res, fname="FigB2_bavi_snapshots.png"):
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 4.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, t in zip(axes.flat, snapshot_times):
        draw_terrain(ax, L)
        N = np.ma.masked_less(res["snaps"][t], SHOW_MIN)
        im = ax.imshow(N, extent=EXTENT, origin="lower", cmap=POP_CMAP,
                       vmin=0, vmax=K0, alpha=0.85)
        mark_start(ax)
        ax.set_title(f"t = {t} years", fontsize=11)
        ax.label_outer()
    label_features(axes[0, 0], L)
    fig.colorbar(im, ax=axes, shrink=0.85).set_label("Population density $N$")
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    print("saved", fname)


def fig_arrival(L, res, fname="FigB3_bavi_arrival_speed.png"):
    c_flat = 2 * math.sqrt(r0 * D0)
    arrival = res["arrival"]
    speed = front_speed(arrival, L)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.0), sharey=True, constrained_layout=True)

    ax = axes[0]
    draw_terrain(ax, L)
    im = ax.imshow(np.ma.masked_invalid(arrival), extent=EXTENT, origin="lower",
                   cmap="Oranges_r", vmin=0, vmax=T_end, alpha=0.85)
    cs = ax.contour(X, Y, np.where(np.isnan(arrival), T_end + 1, arrival),
                    levels=np.arange(5, T_end + 1, 5), colors="#222222", linewidths=0.6)
    ax.clabel(cs, fmt="%d y", fontsize=8)
    mark_start(ax)
    ax.set_title("Arrival time", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.9).set_label("years")

    ax = axes[1]
    draw_terrain(ax, L)
    im = ax.imshow(np.ma.masked_invalid(speed / c_flat), extent=EXTENT, origin="lower",
                   cmap="PuOr", norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2), alpha=0.9)
    mark_start(ax)
    label_features(ax, L)
    ax.set_title(f"Front speed / $c$  ($c$ = {c_flat:.2f} km/year)", fontsize=11)
    ax.set_ylabel("")
    fig.colorbar(im, ax=ax, shrink=0.9).set_label("speed ratio")

    fig.savefig(fname, dpi=250)
    plt.close(fig)
    print("saved", fname)


def animate(L):
    faces = face_coefficients(L)
    dt = stable_dt(L, faces)
    state = {"N": initial_condition(), "t": 0.0}
    steps_per_frame = 60

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_terrain(ax, L)
    im = ax.imshow(np.ma.masked_less(state["N"], SHOW_MIN), extent=EXTENT, origin="lower",
                   cmap=POP_CMAP, vmin=0, vmax=K0, alpha=0.85)
    mark_start(ax)
    label_features(ax, L)
    fig.colorbar(im, ax=ax).set_label("Population density N(x,y,t)")
    title = ax.set_title("t = 0")

    def update(frame):
        N = state["N"]
        for _ in range(steps_per_frame):
            N = np.maximum(N + dt * rhs(N, L, faces), 0.0)
        state["N"] = N
        state["t"] += steps_per_frame * dt
        im.set_data(np.ma.masked_less(N, SHOW_MIN))
        title.set_text(f"Ba Vì,  t = {state['t']:.1f} years")
        return im, title

    frames = int(T_end / (steps_per_frame * dt))
    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    L = build_landscape()
    print(f"grid {Ny} x {Nx} cells of {1000 * dx:.0f} m x {1000 * dy:.0f} m")
    j0 = int(round((y_init - y[0]) / dy))
    i0 = int(round((x_init - x[0]) / dx))
    print(f"start: z = {Z[j0, i0]:.0f} m, slope = {L['slope'][j0, i0]:.3f}, "
          f"river at start: {bool(L['river'][j0, i0])}")

    res = simulate(L)
    c_flat = 2 * math.sqrt(r0 * D0)
    print(f"Fisher-KPP front speed on flat land: c = 2 sqrt(r0 D0) = {c_flat:.2f} km/year")
    reached = ~np.isnan(res["arrival"]) & ~L["river"]
    print(f"land reached after {T_end:g} years: {100 * reached.sum() / (~L['river']).sum():.0f}%")

    # typical front speed on flat land and on steep slopes (median, so that the
    # spikes where two fronts meet do not count)
    ratio = front_speed(res["arrival"], L) / c_flat
    flat = (L["slope"] < 0.05) & ~L["river"]
    steep = (L["slope"] > 0.3) & ~L["river"]
    print(f"front speed / c (median): flat land {np.nanmedian(ratio[flat]):.2f}, "
          f"steep slopes {np.nanmedian(ratio[steep]):.2f}")
    summit = np.unravel_index(np.argmax(Z), Z.shape)
    print(f"summit: z = {Z[summit]:.0f} m, K = {L['K'][summit]:.1f} "
          f"(arrival threshold {arrival_threshold:g})")
    print(f"river cells: {100 * L['river'].mean():.1f}% of the grid")

    fig_landscape(L)
    fig_snapshots(L, res)
    fig_arrival(L, res)

    if args.animate:
        animate(L)
