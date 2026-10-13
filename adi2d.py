"""
ADI (Alternating Direction Implicit) method for the 2D Fisher-KPP equation,
compared with the explicit Euler method.

    dN/dt = Dx N_xx + Dy N_yy + r N (1 - N/K),   no-flux (Neumann) boundary

One time step (Strang splitting):
  1. reaction, half step dt/2   -> exact solution of the logistic ODE
  2. diffusion, full step dt    -> ADI (Peaceman-Rachford):
         (I - dt/2 Dx Lx) N*      = (I + dt/2 Dy Ly) N^n     implicit in x
         (I - dt/2 Dy Ly) N^{n+1} = (I + dt/2 Dx Lx) N*      implicit in y
  3. reaction, half step dt/2   -> exact solution of the logistic ODE

Each half step only needs 1D tridiagonal systems (one per grid row or column).
The scheme is unconditionally stable and second-order accurate in time, so dt
is limited by accuracy only, not by dt <= dx^2 / (2 (Dx + Dy)).

Both methods use the same spatial operator: central differences with the
Neumann condition through a ghost point (N_{-1} = N_1). The benchmark therefore
compares the time stepping only.

Run:  python adi2d.py   -> prints the benchmark and saves Fig6_adi_benchmark.png
"""

import time

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PARAMETERS (same as spatial2dplot.py)
# ============================================================

L = 100.0
Dx, Dy = 4.0, 1.0
r = 0.1
K = 100.0
T = 60.0                        # final time of the benchmark


def make_grid(n):
    x = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, x)    # arrays are indexed [j, i] = [y, x]
    N0 = 80 * np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * 5**2))
    return x[1] - x[0], N0


# ============================================================
# 2. SPATIAL OPERATOR (Neumann through a ghost point)
# ============================================================

def lap_x(N, h):
    """Second difference in x; at the boundary N_{-1} = N_1."""
    out = np.empty_like(N)
    out[:, 1:-1] = N[:, 2:] - 2 * N[:, 1:-1] + N[:, :-2]
    out[:, 0] = 2 * (N[:, 1] - N[:, 0])
    out[:, -1] = 2 * (N[:, -2] - N[:, -1])
    return out / h**2


def lap_y(N, h):
    return lap_x(N.T, h).T


def make_tridiagonal_solver(n, alpha):
    """Thomas algorithm for the tridiagonal system (I - alpha * h^2 * Lap) u = rhs,
    where Lap is the 1D Neumann Laplacian. The matrix is factorized once; the
    returned function solves along axis 0, for every column of rhs at once."""
    sub = np.full(n, -alpha)            # sub-diagonal   (sub[0] unused)
    diag = np.full(n, 1 + 2 * alpha)    # main diagonal
    sup = np.full(n, -alpha)            # super-diagonal (sup[-1] unused)
    sup[0] = -2 * alpha                 # ghost point at the first node
    sub[-1] = -2 * alpha                # ghost point at the last node

    # forward elimination of the matrix (done once)
    w = np.zeros(n)
    d = diag.copy()
    for i in range(1, n):
        w[i] = sub[i] / d[i - 1]
        d[i] = diag[i] - w[i] * sup[i - 1]

    def solve(rhs):
        u = rhs.copy()
        for i in range(1, n):               # forward sweep
            u[i] -= w[i] * u[i - 1]
        u[-1] /= d[-1]
        for i in range(n - 2, -1, -1):      # back substitution
            u[i] = (u[i] - sup[i] * u[i + 1]) / d[i]
        return u

    return solve


def logistic_exact(N, t):
    """Exact solution of dN/dt = r N (1 - N/K) after time t."""
    e = np.exp(r * t)
    return K * N * e / (K + N * (e - 1))


# ============================================================
# 3. SOLVERS
# ============================================================

def solve_explicit(N0, h, dt):
    N = N0.copy()
    for _ in range(round(T / dt)):
        N = N + dt * (Dx * lap_x(N, h) + Dy * lap_y(N, h) + r * N * (1 - N / K))
    return N


def solve_adi(N0, h, dt):
    n = N0.shape[0]
    # the same tridiagonal matrix is used for every row / column and every step
    solve_x = make_tridiagonal_solver(n, dt / 2 * Dx / h**2)
    solve_y = make_tridiagonal_solver(n, dt / 2 * Dy / h**2)

    N = N0.copy()
    for _ in range(round(T / dt)):
        N = logistic_exact(N, dt / 2)
        # half step 1: implicit in x (one tridiagonal system per row)
        N_star = solve_x((N + dt / 2 * Dy * lap_y(N, h)).T).T
        # half step 2: implicit in y (one tridiagonal system per column)
        N = solve_y(N_star + dt / 2 * Dx * lap_x(N_star, h))
        N = logistic_exact(N, dt / 2)
    return N


def timed(solver, N0, h, dt):
    """Best of several runs (5 for short runs, 2 for long ones)."""
    best = np.inf
    runs = 0
    while runs < 2 or (runs < 5 and best < 1.0):
        t0 = time.perf_counter()
        N = solver(N0, h, dt)
        best = min(best, time.perf_counter() - t0)
        runs += 1
    return N, best


# ============================================================
# 4. BENCHMARK
# ============================================================
# Error = max |N - N_ref| / K at t = T. The reference is ADI with dt = 0.01;
# explicit Euler with a tiny dt converges to the same solution.

grids = [101, 201]
adi_dts = [4.0, 2.0, 1.0, 0.5, 0.25]
explicit_factors = [0.8, 0.4, 0.2]          # dt = factor * stability limit

results = {}
for n in grids:
    h, N0 = make_grid(n)
    dt_max = 1 / (2 * (Dx + Dy) / h**2)
    N_ref = solve_adi(N0, h, 0.01)
    print(f"\nGrid {n} x {n}  (dx = {h:g}, explicit limit dt_max = {dt_max:g})")

    rows = {"Explicit Euler": [], "ADI": []}
    for dt in [f * dt_max for f in explicit_factors]:
        N, sec = timed(solve_explicit, N0, h, dt)
        rows["Explicit Euler"].append((dt, sec, 100 * np.abs(N - N_ref).max() / K))
    for dt in adi_dts:
        N, sec = timed(solve_adi, N0, h, dt)
        rows["ADI"].append((dt, sec, 100 * np.abs(N - N_ref).max() / K))

    for method, data in rows.items():
        for dt, sec, err in data:
            print(f"  {method:15s} dt = {dt:<7.4g} time = {sec:6.3f} s   error = {err:.4f} % of K")
    results[n] = (h, rows)

# ============================================================
# 5. PLOT: error vs. computing time
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#222222",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
})

style = {
    "Explicit Euler": dict(color="#A02020", marker="o"),
    "ADI": dict(color="#2B6CB0", marker="s"),
}

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True, sharey=True)

for ax, n in zip(axes, grids):
    h, rows = results[n]
    for method, data in rows.items():
        dts, secs, errs = zip(*data)
        ax.loglog(secs, errs, linewidth=2, markersize=6, label=method,
                  **style[method])
    # label the fastest stable explicit run and the ADI run with dt = 2
    for method, k, dx_pt, ha in [("Explicit Euler", 0, -8, "right"), ("ADI", 1, 8, "left")]:
        dt, sec, err = rows[method][k]
        ax.annotate(rf"$\Delta t$ = {dt:g}", (sec, err), textcoords="offset points",
                    xytext=(dx_pt, 0), ha=ha, va="center", fontsize=9, color="#222222")

    ax.set_title(f"Grid {n}$\\times${n}  ($\\Delta x$ = {h:g})", fontsize=11, color="#222222")
    ax.set_xlabel("Computing time (s)")
    ax.set_xlim(0.01, 10)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(color="#e6e6e6", linewidth=0.8, which="major")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Max error (% of K)")
axes[0].legend(frameon=False, fontsize=9, loc="center right")

plt.tight_layout()
plt.savefig("Fig6_adi_benchmark.png", dpi=250)
plt.show()
