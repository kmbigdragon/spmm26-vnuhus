import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. PARAMETERS
# ============================================================

Lx = 100
Ly = 100

Dx = 4
Dy = 1
r = 0.1
K = 100.0

# Initial population
N0 = 80
x0 = 50
y0 = 50
sigma = 5

# ============================================================
# 2. GRID
# ============================================================

Nx = 101
Ny = 101

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

X, Y = np.meshgrid(x, y)

# ============================================================
# 3. TIME
# ============================================================

dt = 0.8 / (
    2 * Dx / dx**2 +
    2 * Dy / dy**2
)

T = 100

Nt = int(T / dt)

print("dt =", dt)
print("Number of steps =", Nt)

# ============================================================
# 4. INITIAL CONDITION
# ============================================================

N = N0 * np.exp(
    -(
        (X - x0)**2
        +
        (Y - y0)**2
    ) / (2 * sigma**2)
)

# ============================================================
# 5. SAVE SNAPSHOTS
# ============================================================

snapshot_times = [
    0,
    10,
    20,
    40,
    60,
    80,
    100
]

snapshots = []

current_time = 0

snapshot_index = 0

# Save initial condition
snapshots.append(N.copy())

next_snapshot = 1

# ============================================================
# 6. SOLVE PDE
# ============================================================

for step in range(1, Nt + 1):

    # Second derivative in x
    N_xx = (
        N[1:-1, 2:]
        - 2 * N[1:-1, 1:-1]
        + N[1:-1, :-2]
    ) / dx**2

    # Second derivative in y
    N_yy = (
        N[2:, 1:-1]
        - 2 * N[1:-1, 1:-1]
        + N[:-2, 1:-1]
    ) / dy**2

    # Logistic growth
    growth = (
        r
        * N[1:-1, 1:-1]
        * (1 - N[1:-1, 1:-1] / K)
    )

    # Update
    N_new = N.copy()

    N_new[1:-1, 1:-1] = (
        N[1:-1, 1:-1]
        + dt * (
            Dx * N_xx
            + Dy * N_yy
            + growth
        )
    )

    # --------------------------------------------------------
    # Neumann boundary conditions
    # --------------------------------------------------------

    N_new[:, 0] = N_new[:, 1]
    N_new[:, -1] = N_new[:, -2]

    N_new[0, :] = N_new[1, :]
    N_new[-1, :] = N_new[-2, :]

    N = N_new

    current_time = step * dt

    # Save snapshot when passing requested time
    if (
        next_snapshot < len(snapshot_times)
        and current_time >= snapshot_times[next_snapshot]
    ):
        snapshots.append(N.copy())
        next_snapshot += 1


# ============================================================
# 7. 3D CONTOUR PLOT
# ============================================================

fig = plt.figure(figsize=(11, 9))

ax = fig.add_subplot(111, projection="3d")

# Number of contour levels
levels = np.linspace(10, 90, 9)

# Vertical separation between time planes
time_scale = 0.8

for i, (t, N_snapshot) in enumerate(
    zip(snapshot_times, snapshots)
):

    # Z coordinate represents time
    Z = np.ones_like(X) * t * time_scale

    # Draw contour on horizontal plane
    ax.contour(
        X,
        Y,
        N_snapshot,
        levels=levels,
        zdir="z",
        offset=t * time_scale,
        cmap=cm.viridis
    )

# ============================================================
# 8. AXES
# ============================================================

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Time")

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)

ax.set_zlim(
    0,
    max(snapshot_times) * time_scale
)

ax.set_title(
    "Spatial Population Spread Over Time"
)

plt.tight_layout()

plt.show()