import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. PARAMETERS
# ============================================================

# Size of the spatial domain
Lx = 100
Ly = 100

# Diffusion coefficients
Dx = 10
Dy = 5

# Logistic growth parameters
r = 0.1
K = 100.0

# Initial population parameters
N0 = 80.0
x0 = 50.0
y0 = 50.0
sigma = 5.0

# ============================================================
# 2. SPATIAL GRID
# ============================================================

Nx = 101
Ny = 101

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

X, Y = np.meshgrid(x, y)

# ============================================================
# 3. TIME STEP
# ============================================================

# Stability condition for explicit finite difference
dt_diffusion = 0.5 / (
    Dx / dx**2 +
    Dy / dy**2
)

dt = 0.8 * dt_diffusion

T = 100

# Number of numerical steps
Nt = int(T / dt)

print("dx =", dx)
print("dy =", dy)
print("dt =", dt)
print("Number of time steps =", Nt)

# ============================================================
# 4. INITIAL CONDITION
# ============================================================

# Gaussian population concentrated at the center
N = N0 * np.exp(
    -((X - x0)**2 + (Y - y0)**2)
    / (2 * sigma**2)
)

# ============================================================
# 5. FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

image = ax.imshow(
    N,
    extent=[0, Lx, 0, Ly],
    origin="lower",
    interpolation="bilinear",
    vmin=0,
    vmax=K,
    aspect="equal"
)

ax.set_xlabel("x")
ax.set_ylabel("y")

title = ax.set_title("2D Spatial Population Growth")

# Colorbar
cbar = fig.colorbar(image, ax=ax)
cbar.set_label("Population density N(x,y,t)")

# ============================================================
# 6. UPDATE FUNCTION
# ============================================================

def update(frame):

    global N

    # Number of numerical steps between two animation frames
    steps_per_frame = 10

    for _ in range(steps_per_frame):

        # ----------------------------------------------------
        # Second derivative in x
        # ----------------------------------------------------

        N_xx = (
            N[1:-1, 2:]
            - 2 * N[1:-1, 1:-1]
            + N[1:-1, :-2]
        ) / dx**2

        # ----------------------------------------------------
        # Second derivative in y
        # ----------------------------------------------------

        N_yy = (
            N[2:, 1:-1]
            - 2 * N[1:-1, 1:-1]
            + N[:-2, 1:-1]
        ) / dy**2

        # ----------------------------------------------------
        # Logistic growth
        # ----------------------------------------------------

        growth = (
            r
            * N[1:-1, 1:-1]
            * (1 - N[1:-1, 1:-1] / K)
        )

        # ----------------------------------------------------
        # PDE
        # ----------------------------------------------------

        N_new = N.copy()

        N_new[1:-1, 1:-1] = (
            N[1:-1, 1:-1]
            + dt * (
                Dx * N_xx
                + Dy * N_yy
                + growth
            )
        )

        # ----------------------------------------------------
        # Neumann boundary condition
        # No population flux across boundary
        # ----------------------------------------------------

        N_new[:, 0] = N_new[:, 1]
        N_new[:, -1] = N_new[:, -2]

        N_new[0, :] = N_new[1, :]
        N_new[-1, :] = N_new[-2, :]

        N = N_new

    # Update heatmap
    image.set_data(N)

    # Current time
    current_time = frame * steps_per_frame * dt

    title.set_text(
        f"2D Spatial Population Growth\n"
        f"t = {current_time:.2f}    "
        f"Dx = {Dx}    Dy = {Dy}"
    )

    return image, title


# ============================================================
# 7. CREATE ANIMATION
# ============================================================

frames = Nt // 10

animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=30,
    blit=True
)

plt.tight_layout()
plt.show()