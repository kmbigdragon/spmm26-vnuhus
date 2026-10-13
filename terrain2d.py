"""
Spatial population growth on a heterogeneous landscape (single species).

Model (reaction - diffusion - taxis):

    dN/dt = div( D(x) grad N )                (movement, slowed by terrain)
          - div( chi(x) N grad A(x) )         (drift toward attractive habitat)
          + r(x) N (1 - N / K(x))             (local logistic growth)

on Omega = [0,100]^2 minus the lake, with no-flux boundary conditions on the
outer boundary AND on the lake shore.

Landscape features (synthetic, elevation in metres):
  * mountain ridge  -> steep slope lowers mobility D, high altitude lowers K
  * mountain pass   -> a low gap in the ridge where crossing is easy
  * river           -> narrow strip with very low D and negative r (risky crossing)
  * ford            -> shallow stretch of the river that is easy to cross
  * river valley    -> fertile lowland: higher K and attracts the population
  * lake            -> hole in the domain (no-flux boundary around it)

Numerics: cell-centred finite volumes + explicit Euler.
  * diffusive flux on each cell face uses the harmonic mean of D
    (a face next to a lake cell gets D = 0, so the shore is automatically no-flux)
  * taxis flux uses first-order upwinding
  * dt is chosen from the diffusion + advection (CFL) stability limits

Run:  python terrain2d.py              -> saves FigT1..FigT3 png files
      python terrain2d.py --animate    -> also shows a live animation
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. PARAMETERS
# ============================================================

Lx, Ly = 100.0, 100.0          # domain size
Nx, Ny = 200, 200              # number of cells

D0 = 2.0                       # mobility on flat land
r0 = 0.1                       # intrinsic growth rate
K0 = 100.0                     # reference carrying capacity
chi0 = 2.0                     # strength of attraction toward good habitat

s0 = 40.0                      # slope scale (m per unit length): D = D0 exp(-slope/s0)
z_D = 500.0                    # above this altitude movement gets harder too
z_D_scale = 300.0
z_c = 700.0                    # altitude scale for habitat quality
valley_bonus = 0.6             # extra habitat quality in the river valley

river_halfwidth = 1.2
river_D_factor = 0.005         # mobility inside the river (fraction of local D)
river_r = -0.2                 # mortality while crossing the river
ford_y = (74.0, 84.0)          # y-range where the river is shallow
ford_D_factor = 0.5

lake_centre = (84.0, 80.0)
lake_axes = (9.0, 6.0)

# initial population (Gaussian, like the homogeneous model)
N_init, x_init, y_init, sigma_init = 80.0, 12.0, 15.0, 3.0

T_end = 300.0
arrival_threshold = 0.2 * K0   # "the population has arrived" when N >= this
snapshot_times = [0, 40, 80, 140, 200, 300]

DIFFUSION = "fickian"          # "fickian": div(D grad N); "ecological": lap(D N)

# ============================================================
# 2. GRID (cell centres)
# ============================================================

dx = Lx / Nx
dy = Ly / Ny
x = (np.arange(Nx) + 0.5) * dx
y = (np.arange(Ny) + 0.5) * dy
X, Y = np.meshgrid(x, y)       # arrays are indexed [j, i] = [y, x]


# ============================================================
# 3. LANDSCAPE
# ============================================================

def river_centre(yy):
    """x-coordinate of the river's centre line (a meander)."""
    return 35.0 + 6.0 * np.sin(2 * np.pi * yy / 60.0)


def build_landscape():
    """Return a dict with elevation, D, r, K, A, chi and the feature masks."""

    # --- elevation z(x, y) ---------------------------------------------
    z = 150.0 + 1.5 * X                                   # land rises to the east

    ridge_x = 65.0 + 4.0 * np.sin(2 * np.pi * Y / 80.0)
    ridge = 900.0 * np.exp(-(X - ridge_x) ** 2 / (2 * 6.0 ** 2))
    pass_gap = 1.0 - 0.8 * np.exp(-(Y - 55.0) ** 2 / (2 * 5.0 ** 2))
    z += ridge * pass_gap                                 # ridge with a pass at y ~ 55

    dist_river = X - river_centre(Y)
    z -= 100.0 * np.exp(-dist_river ** 2 / (2 * 10.0 ** 2))   # river valley
    z = np.maximum(z, 0.0)

    # --- masks ------------------------------------------------------------
    river = np.abs(dist_river) <= river_halfwidth
    ford = river & (Y >= ford_y[0]) & (Y <= ford_y[1])
    lake = ((X - lake_centre[0]) / lake_axes[0]) ** 2 \
         + ((Y - lake_centre[1]) / lake_axes[1]) ** 2 <= 1.0

    # --- habitat quality h(x, y): high in the valley, low on the ridge ---
    h = np.exp(-(z / z_c) ** 2) \
        * (1.0 + valley_bonus * np.exp(-dist_river ** 2 / (2 * 8.0 ** 2)))

    # --- mobility D(x, y): slow on steep slopes --------------------------
    gz_y, gz_x = np.gradient(z, dy, dx)
    slope = np.hypot(gz_x, gz_y)
    D = D0 * np.exp(-slope / s0) \
          * np.exp(-np.maximum(z - z_D, 0.0) / z_D_scale)

    # --- growth rate r(x, y) and carrying capacity K(x, y) ---------------
    r = np.full_like(X, r0)
    K = K0 * h

    # river: hard and risky to cross, except at the ford
    D[river & ~ford] *= river_D_factor
    r[river & ~ford] = river_r
    D[ford] *= ford_D_factor
    r[ford] = 0.0

    # lake: removed from the domain
    D[lake] = 0.0
    r[lake] = 0.0

    # --- attraction potential A and taxis coefficient chi ----------------
    A = h.copy()
    chi = chi0 * D / D0            # animals drift as fast as they can move

    return dict(z=z, slope=slope, D=D, r=r, K=K, A=A, chi=chi,
                river=river, ford=ford, lake=lake)


def build_homogeneous():
    """The original model: constant D, r, K, no obstacles, no attraction."""
    ones = np.ones_like(X)
    zero = np.zeros_like(X, dtype=bool)
    return dict(z=0 * ones, slope=0 * ones, D=D0 * ones, r=r0 * ones,
                K=K0 * ones, A=0 * ones, chi=0 * ones,
                river=zero, ford=zero, lake=zero)


# ============================================================
# 4. SOLVER
# ============================================================

def harmonic_mean(a, b):
    s = a + b
    return np.where(s > 0, 2 * a * b / np.where(s > 0, s, 1.0), 0.0)


def face_coefficients(L):
    """Diffusivity and taxis velocity on the interior cell faces."""
    D, chi, A = L["D"], L["chi"], L["A"]

    # x-faces between cell (j, i-1) and (j, i): shape (Ny, Nx-1)
    Dfx = harmonic_mean(D[:, :-1], D[:, 1:])
    chifx = harmonic_mean(chi[:, :-1], chi[:, 1:])
    vfx = chifx * (A[:, 1:] - A[:, :-1]) / dx

    # y-faces between cell (j-1, i) and (j, i): shape (Ny-1, Nx)
    Dfy = harmonic_mean(D[:-1, :], D[1:, :])
    chify = harmonic_mean(chi[:-1, :], chi[1:, :])
    vfy = chify * (A[1:, :] - A[:-1, :]) / dy

    # a face is open only if animals can move on both sides
    open_x = Dfx > 0
    open_y = Dfy > 0
    return Dfx, vfx * open_x, open_x, Dfy, vfy * open_y, open_y


def stable_dt(L, safety=0.8):
    Dfx, vfx, _, Dfy, vfy, _ = face_coefficients(L)
    Dmax = max(Dfx.max(), Dfy.max(), L["D"].max())
    rate = 2 * Dmax / dx ** 2 + 2 * Dmax / dy ** 2 \
         + np.abs(vfx).max() / dx + np.abs(vfy).max() / dy \
         + np.abs(L["r"]).max()
    return safety / rate


def rhs(N, L, faces):
    """Right-hand side dN/dt of the PDE (finite-volume form)."""
    Dfx, vfx, open_x, Dfy, vfy, open_y = faces
    D = L["D"]

    # --- diffusive fluxes -------------------------------------------------
    if DIFFUSION == "fickian":            # F = -D grad N
        Fx = -Dfx * (N[:, 1:] - N[:, :-1]) / dx
        Fy = -Dfy * (N[1:, :] - N[:-1, :]) / dy
    else:                                 # ecological: F = -grad(D N)
        DN = D * N
        Fx = -1.0 * open_x * (DN[:, 1:] - DN[:, :-1]) / dx
        Fy = -1.0 * open_y * (DN[1:, :] - DN[:-1, :]) / dy

    # --- taxis (advective) fluxes, upwind --------------------------------
    Fx = Fx + vfx * np.where(vfx > 0, N[:, :-1], N[:, 1:])
    Fy = Fy + vfy * np.where(vfy > 0, N[:-1, :], N[1:, :])

    # --- no-flux on the outer boundary: pad with zero fluxes -------------
    Fx = np.pad(Fx, ((0, 0), (1, 1)))
    Fy = np.pad(Fy, ((1, 1), (0, 0)))

    divF = (Fx[:, 1:] - Fx[:, :-1]) / dx + (Fy[1:, :] - Fy[:-1, :]) / dy

    # --- reaction ---------------------------------------------------------
    # habitable cells (r > 0): logistic growth r N (1 - N/K)
    # hostile cells  (r <= 0): pure mortality r N
    # (logistic with r < 0 would turn into explosive growth once N > K)
    r = L["r"]
    growth = np.where(r > 0, r * N * (1.0 - N / L["K"]), r * N)

    return -divF + growth


def initial_condition(L):
    N = N_init * np.exp(-((X - x_init) ** 2 + (Y - y_init) ** 2)
                        / (2 * sigma_init ** 2))
    N[L["lake"]] = 0.0
    return N


def simulate(L, T=T_end, snaps=snapshot_times, verbose=True):
    faces = face_coefficients(L)
    dt = stable_dt(L)
    Nt = int(np.ceil(T / dt))
    dt = T / Nt
    if verbose:
        print(f"dt = {dt:.4f}, steps = {Nt}")

    N = initial_condition(L)
    arrival = np.full(N.shape, np.nan)
    arrival[N >= arrival_threshold] = 0.0

    saved = {0: N.copy()} if 0 in snaps else {}
    next_snaps = [s for s in snaps if s > 0]

    for n in range(1, Nt + 1):
        N = N + dt * rhs(N, L, faces)
        N = np.maximum(N, 0.0)
        N[L["lake"]] = 0.0
        t = n * dt

        newly = (N >= arrival_threshold) & np.isnan(arrival)
        arrival[newly] = t

        while next_snaps and t >= next_snaps[0] - 1e-9:
            saved[next_snaps.pop(0)] = N.copy()

    return dict(N=N, snaps=saved, arrival=arrival, dt=dt)


# ============================================================
# 5. PLOTTING HELPERS
# ============================================================

EXTENT = [0, Lx, 0, Ly]
WATER = "#2a7fd4"


def draw_terrain(ax, L, contours=True):
    """Overlay ridge contours, river and lake on an axis."""
    if contours and L["z"].max() > 0:
        ax.contour(X, Y, L["z"], levels=[400, 600, 800],
                   colors="0.25", linewidths=0.6, alpha=0.7)
    if L["river"].any():
        river_only = L["river"] & ~L["ford"]
        ax.contourf(X, Y, river_only.astype(float), levels=[0.5, 1.5],
                    colors=[WATER])
        ax.contourf(X, Y, L["ford"].astype(float), levels=[0.5, 1.5],
                    colors=[WATER], alpha=0.35)
    if L["lake"].any():
        ax.contourf(X, Y, L["lake"].astype(float), levels=[0.5, 1.5],
                    colors=[WATER])
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal")


def label_features(ax):
    style = dict(fontsize=8, color="0.1",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.text(67, 92, "ridge", ha="center", **style)
    ax.text(73, 55, "pass", ha="left", va="center", **style)
    ax.text(42, 79, "ford", ha="left", va="center", **style)
    ax.text(84, 80, "lake", ha="center", va="center", **style)
    ax.text(36, 3, "river", ha="center", **style)
    ax.plot(x_init, y_init, marker="*", ms=11, color="white", mec="black", mew=0.8)


# ============================================================
# 6. FIGURES
# ============================================================

def fig_landscape(L, fname="FigT1_landscape.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2), constrained_layout=True)

    panels = [
        (L["z"], "YlOrBr", "Elevation $z(x,y)$ [m]"),
        (L["D"] / D0, "Greys_r", "Mobility $D(x,y)/D_0$"),
        (L["K"], "YlGn", "Carrying capacity $K(x,y)$"),
        (L["A"], "YlGn", "Attraction $A(x,y)$ and drift direction"),
    ]
    for ax, (field, cmap, title) in zip(axes.flat, panels):
        f = np.ma.masked_where(L["lake"], field)
        im = ax.imshow(f, extent=EXTENT, origin="lower", cmap=cmap)
        draw_terrain(ax, L, contours=(cmap != "YlOrBr"))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    label_features(axes[0, 0])

    # drift arrows chi grad A (subsampled)
    ax = axes[1, 1]
    gA_y, gA_x = np.gradient(L["A"], dy, dx)
    vx, vy = L["chi"] * gA_x, L["chi"] * gA_y
    k = 10
    ax.quiver(X[::k, ::k], Y[::k, ::k], vx[::k, ::k], vy[::k, ::k],
              color="0.15", scale=2.5, width=0.004)

    fig.suptitle("Synthetic landscape: how terrain enters the model", fontsize=13)
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print("saved", fname)


def fig_snapshots(L, res, fname="FigT2_snapshots.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), constrained_layout=True)
    for ax, t in zip(axes.flat, snapshot_times):
        N = np.ma.masked_where(L["lake"], res["snaps"][t])
        im = ax.imshow(N, extent=EXTENT, origin="lower", cmap="viridis",
                       vmin=0, vmax=K0 * (1 + valley_bonus))
        draw_terrain(ax, L)
        ax.set_title(f"t = {t}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    label_features(axes[0, 0])
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("Population density $N(x,y,t)$")
    fig.suptitle("Population spreading over the landscape", fontsize=13)
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print("saved", fname)


def fig_arrival(L_h, res_h, L_t, res_t, fname="FigT3_arrival_time.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), constrained_layout=True)
    vmax = T_end
    levels = np.arange(25, T_end + 1, 25)
    for ax, L, res, title in [
        (axes[0], L_h, res_h, "Homogeneous model (constant D, r, K)"),
        (axes[1], L_t, res_t, "Terrain model"),
    ]:
        arr = np.ma.masked_invalid(np.where(L["lake"], np.nan, res["arrival"]))
        im = ax.imshow(arr, extent=EXTENT, origin="lower", cmap="magma_r",
                       vmin=0, vmax=vmax)
        cs = ax.contour(X, Y, arr.filled(np.nan), levels=levels,
                        colors="white", linewidths=0.7)
        ax.clabel(cs, fmt="%d", fontsize=7)
        draw_terrain(ax, L)
        ax.plot(x_init, y_init, marker="*", ms=11, color="white", mec="black", mew=0.8)
        not_reached = np.isnan(res["arrival"]) & ~L["lake"]
        ax.set_title(f"{title}\nnot reached by t={T_end:g}: "
                     f"{100 * not_reached.mean():.0f}% of area", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label(f"Arrival time (first t with N ≥ {arrival_threshold:g})")
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print("saved", fname)


def animate(L):
    faces = face_coefficients(L)
    dt = stable_dt(L)
    N = initial_condition(L)
    steps_per_frame = 40
    state = {"N": N, "t": 0.0}

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.ma.masked_where(L["lake"], N), extent=EXTENT, origin="lower",
                   cmap="viridis", vmin=0, vmax=K0 * (1 + valley_bonus))
    draw_terrain(ax, L)
    label_features(ax)
    fig.colorbar(im, ax=ax).set_label("Population density N(x,y,t)")
    title = ax.set_title("t = 0")

    def update(frame):
        N = state["N"]
        for _ in range(steps_per_frame):
            N = np.maximum(N + dt * rhs(N, L, faces), 0.0)
            N[L["lake"]] = 0.0
        state["N"] = N
        state["t"] += steps_per_frame * dt
        im.set_data(np.ma.masked_where(L["lake"], N))
        title.set_text(f"Population on a landscape,  t = {state['t']:.1f}")
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
    parser.add_argument("--diffusion", choices=["fickian", "ecological"],
                        default="fickian")
    args = parser.parse_args()
    DIFFUSION = args.diffusion

    L_terrain = build_landscape()
    L_homog = build_homogeneous()

    print("Terrain model:")
    res_terrain = simulate(L_terrain)
    print("Homogeneous model:")
    res_homog = simulate(L_homog)

    c_theory = 2 * np.sqrt(r0 * D0)
    print(f"Fisher-KPP front speed on flat land: c = 2 sqrt(r D) = {c_theory:.3f}")

    fig_landscape(L_terrain)
    fig_snapshots(L_terrain, res_terrain)
    fig_arrival(L_homog, res_homog, L_terrain, res_terrain)

    if args.animate:
        animate(L_terrain)
