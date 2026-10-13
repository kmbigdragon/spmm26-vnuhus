"""
Finite element method (FEM) for the Fisher-KPP equation, compared with the FDM.

    dN/dt = Dx N_xx + Dy N_yy + r N (1 - N/K),   no-flux (Neumann) boundary

FEM ingredients
  * piecewise-linear "hat" functions in 1D, bilinear (Q1) elements in 2D,
    built on the same nodes as the FDM grid
  * mass matrix M and stiffness matrix S. The Neumann condition is a natural
    boundary condition of the weak form, so nothing is imposed at the boundary.
  * semi-implicit Euler in time (diffusion implicit, reaction explicit):
        (M + dt S) N^{n+1} = M (N^n + dt f(N^n))
    The diffusion part has no stability limit, so dt can be much larger
    than in the explicit FDM.

The FDM solvers below are the same schemes as in spatial1dplot.py and
spatial2dplot.py.

Run:  python fem_vs_fdm.py   -> saves Fig5_fem_vs_fdm.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PARAMETERS (same as spatial1dplot.py / spatial2dplot.py)
# ============================================================

r = 0.1
K = 100.0

# 1D
L = 100.0
D = 1.0
Nx1 = 201
x1 = np.linspace(0, L, Nx1)
dx1 = x1[1] - x1[0]
N0_1d = 100 * np.exp(-(x1 - 50)**2 / (2 * 5**2))

# 2D
Lx, Ly = 100.0, 100.0
Dx, Dy = 4.0, 1.0
Nx2, Ny2 = 101, 101
x2 = np.linspace(0, Lx, Nx2)
y2 = np.linspace(0, Ly, Ny2)
dx2 = x2[1] - x2[0]
dy2 = y2[1] - y2[0]
X, Y = np.meshgrid(x2, y2)                  # arrays are indexed [j, i] = [y, x]
N0_2d = 80 * np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * 5**2))

# time steps
dt_fdm_1d = 0.2 * dx1**2 / D                                # 0.05 (limit 0.125)
dt_fdm_2d = 0.8 / (2 * Dx / dx2**2 + 2 * Dy / dy2**2)       # 0.08 (limit 0.1)
dt_fem = 0.5                                                # no diffusion limit

times_1d = [0, 25, 50, 75]
times_2d = [20, 40, 60]


def growth(N):
    return r * N * (1 - N / K)


def run(step, N, dt, times):
    """Advance N with `step` and return copies of N at the requested times."""
    steps = [round(t / dt) for t in times]
    out = {}
    for n in range(max(steps) + 1):
        if n in steps:
            out[n] = N.copy()
        N = step(N)
    return [out[n] for n in steps]


# ============================================================
# 2. FEM MATRICES (1D, piecewise-linear elements)
# ============================================================

def fem_matrices_1d(n, h):
    """Mass matrix M_kl = int phi_k phi_l, stiffness S_kl = int phi_k' phi_l'."""
    M = np.zeros((n, n))
    S = np.zeros((n, n))
    M_elem = h / 6 * np.array([[2, 1], [1, 2]])
    S_elem = 1 / h * np.array([[1, -1], [-1, 1]])
    for e in range(n - 1):                  # assemble element by element
        idx = np.ix_([e, e + 1], [e, e + 1])
        M[idx] += M_elem
        S[idx] += S_elem
    return M, S


# ============================================================
# 3. SOLVERS
# ============================================================

# ---------- FDM 1D (explicit Euler) ----------
def fdm_1d_step(N):
    N_xx = np.zeros_like(N)
    N_xx[1:-1] = (N[2:] - 2 * N[1:-1] + N[:-2]) / dx1**2
    N_new = N + dt_fdm_1d * (D * N_xx + growth(N))
    N_new[0] = N_new[1]                     # Neumann boundary condition
    N_new[-1] = N_new[-2]
    return N_new


# ---------- FEM 1D (semi-implicit Euler) ----------
M1, S1 = fem_matrices_1d(Nx1, dx1)
A1 = M1 + dt_fem * D * S1                   # same matrix every step


def fem_1d_step(N):
    rhs = M1 @ (N + dt_fem * growth(N))
    return np.linalg.solve(A1, rhs)         # no boundary treatment needed


# ---------- FDM 2D (explicit Euler) ----------
def fdm_2d_step(N):
    N_xx = (N[1:-1, 2:] - 2 * N[1:-1, 1:-1] + N[1:-1, :-2]) / dx2**2
    N_yy = (N[2:, 1:-1] - 2 * N[1:-1, 1:-1] + N[:-2, 1:-1]) / dy2**2
    N_new = N.copy()
    N_new[1:-1, 1:-1] += dt_fdm_2d * (Dx * N_xx + Dy * N_yy + growth(N[1:-1, 1:-1]))
    N_new[:, 0] = N_new[:, 1]               # Neumann boundary condition
    N_new[:, -1] = N_new[:, -2]
    N_new[0, :] = N_new[1, :]
    N_new[-1, :] = N_new[-2, :]
    return N_new


# ---------- FEM 2D (bilinear Q1 elements, semi-implicit Euler) ----------
# On a rectangular grid the Q1 basis is phi_i(x) * psi_j(y), so every 2D
# matrix is built from the 1D ones:
#     mass       M2 = My (x) Mx
#     stiffness  S2 = Dx My (x) Sx + Dy Sy (x) Mx
# With N stored as an (Ny, Nx) array, M2 N <-> My @ N @ Mx, etc.
# The system (M2 + dt S2) N^{n+1} = rhs is solved exactly with the
# generalized eigenvectors of the 1D pairs (S, M) ("fast diagonalization"):
#     V^T M V = I,   V^T S V = diag(lam).
Mx, Sx = fem_matrices_1d(Nx2, dx2)
My, Sy = fem_matrices_1d(Ny2, dy2)


def generalized_eig(S, M):
    L_inv = np.linalg.inv(np.linalg.cholesky(M))
    lam, W = np.linalg.eigh(L_inv @ S @ L_inv.T)
    return lam, L_inv.T @ W


lam_x, Vx = generalized_eig(Sx, Mx)
lam_y, Vy = generalized_eig(Sy, My)
denom = 1 + dt_fem * (Dx * lam_x[None, :] + Dy * lam_y[:, None])


def fem_2d_step(N):
    rhs = My @ (N + dt_fem * growth(N)) @ Mx
    N_hat = (Vy.T @ rhs @ Vx) / denom
    return Vy @ N_hat @ Vx.T


# ============================================================
# 4. RUN
# ============================================================

fdm1 = run(fdm_1d_step, N0_1d, dt_fdm_1d, times_1d)
fem1 = run(fem_1d_step, N0_1d, dt_fem, times_1d)
fdm2 = run(fdm_2d_step, N0_2d, dt_fdm_2d, times_2d)
fem2 = run(fem_2d_step, N0_2d, dt_fem, times_2d)

for name, a, b, dt_a in [("1D", fdm1, fem1, dt_fdm_1d), ("2D", fdm2, fem2, dt_fdm_2d)]:
    diff = max(np.abs(p - q).max() for p, q in zip(a, b)) / K
    steps = round(max(times_1d if name == "1D" else times_2d) / dt_a)
    steps_fem = round(max(times_1d if name == "1D" else times_2d) / dt_fem)
    print(f"{name}: max |FEM - FDM| / K = {100 * diff:.2f} %   "
          f"(FDM dt = {dt_a:g}, {steps} steps;  FEM dt = {dt_fem:g}, {steps_fem} steps)")

# ============================================================
# 5. PLOT
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#222222",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
})

# light -> dark: later times are darker
colors_1d = ["#E3A0A0", "#CC6666", "#A82828", "#700000"]
colors_2d = ["#D98080", "#A82828", "#700000"]

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1.55, 1]}
)

# ---------- 1D: FDM lines, FEM markers ----------
for N_fdm, N_fem, t, c in zip(fdm1, fem1, times_1d, colors_1d):
    ax1.plot(x1, N_fdm, color=c, linewidth=2, label=f"t = {t}")
    ax1.plot(x1[::8], N_fem[::8], "o", color=c, markersize=4,
             markerfacecolor="white", markeredgewidth=1.2)

ax1.plot([], [], color="#555555", linewidth=2, label="FDM")
ax1.plot([], [], "o", color="#555555", markersize=4,
         markerfacecolor="white", markeredgewidth=1.2, label="FEM")
ax1.set_title("1D profiles", fontsize=11, color="#222222")
ax1.set_xlabel("Position x")
ax1.set_ylabel("Density N")
ax1.set_xlim(0, L)
ax1.set_ylim(-5, 150)
ax1.set_yticks([0, 50, 100])
ax1.legend(frameon=False, ncol=3, fontsize=9, loc="upper center",
           handlelength=1.5, columnspacing=0.8)
ax1.grid(color="#e6e6e6", linewidth=0.8)
ax1.spines[["top", "right"]].set_visible(False)

# ---------- 2D: front N = K/2, FDM solid vs FEM dashed ----------
for N_fdm, N_fem, t, c in zip(fdm2, fem2, times_2d, colors_2d):
    ax2.contour(X, Y, N_fdm, levels=[K / 2], colors=[c], linewidths=2)
    ax2.contour(X, Y, N_fem, levels=[K / 2], colors=["#222222"],
                linewidths=1, linestyles="--")
    # label just above the front on the vertical line x = 50
    y_front = y2[N_fdm[:, Nx2 // 2] >= K / 2].max()
    ax2.text(50, y_front + 1.5, f"t = {t}", ha="center", va="bottom",
             fontsize=9, color="#222222")

ax2.plot([], [], color="#A82828", linewidth=2, label="FDM")
ax2.plot([], [], color="#222222", linewidth=1, linestyle="--", label="FEM")
ax2.legend(frameon=False, ncol=2, fontsize=9, loc="lower center",
           handlelength=1.8, columnspacing=1.0)

ax2.set_title(r"2D front $N = K/2$", fontsize=11, color="#222222")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlim(0, Lx)
ax2.set_ylim(0, Ly)
ax2.set_xticks([0, 50, 100])
ax2.set_yticks([0, 50, 100])
ax2.set_aspect("equal")
for side in ["top", "right"]:
    ax2.spines[side].set_visible(True)

plt.tight_layout()
plt.savefig("Fig5_fem_vs_fdm.png", dpi=250)
plt.show()
