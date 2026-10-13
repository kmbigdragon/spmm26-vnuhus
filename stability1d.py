"""
Stability of the explicit Euler scheme for the 1D Fisher-KPP equation.

    dN/dt = D d2N/dx2 + r N (1 - N/K),   Neumann boundary conditions

The explicit scheme is stable only if  dt <= dx^2 / (2 D).
We run the same problem twice:
  * dt = 0.05  (below the limit)  -> smooth spreading
  * dt = 0.15  (above the limit)  -> oscillations grow and the solution blows up

Run:  python stability1d.py   -> saves Fig4_stability.png
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters (same as spatial1dplot.py)
# =========================
L = 100.0
D = 1.0
r = 0.1
K = 100.0

Nx = 201
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

dt_max = dx**2 / (2 * D)          # stability limit = 0.125
dt_stable = 0.05
dt_unstable = 0.15

snapshot_times = [0, 10, 16.8]

# light -> dark: later times are darker
colors = ["#E3A0A0", "#BD3B3B", "#700000"]
styles = ["--", "-", "-"]


# =========================
# Solver
# =========================
def simulate(dt, times):
    N = 100 * np.exp(-(x - 50)**2 / (2 * 5**2))
    steps = [round(t / dt) for t in times]
    snapshots = {}

    for n in range(max(steps) + 1):
        if n in steps:
            snapshots[n] = N.copy()

        N_xx = np.zeros_like(N)
        N_xx[1:-1] = (N[2:] - 2 * N[1:-1] + N[:-2]) / dx**2

        N = N + dt * D * N_xx + dt * r * N * (1 - N / K)

        # Neumann boundary condition
        N[0] = N[1]
        N[-1] = N[-2]

    return [snapshots[n] for n in steps]


# =========================
# Plot
# =========================
plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#222222",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
})

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)

cases = [
    (dt_stable, rf"$\Delta t = {dt_stable}$  (stable)"),
    (dt_unstable, rf"$\Delta t = {dt_unstable} > {dt_max}$  (unstable)"),
]

for ax, (dt, title) in zip(axes, cases):
    for N_snap, t, c, ls in zip(simulate(dt, snapshot_times), snapshot_times, colors, styles):
        ax.plot(x, N_snap, color=c, linestyle=ls, linewidth=2, label=f"t = {t:g}")

    ax.set_title(title, fontsize=11, color="#222222")
    ax.set_xlabel("Position x")
    ax.set_xlim(0, L)
    ax.set_ylim(-60, 160)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Density N")
axes[0].set_yticks([-50, 0, 50, 100, 150])
axes[0].legend(frameon=False, loc="upper center", ncol=3, fontsize=9,
               handlelength=1.4, columnspacing=0.8)

plt.tight_layout()
plt.savefig("Fig4_stability.png", dpi=200)
plt.show()
