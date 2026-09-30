import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Parameters
# =========================
L = 100          # length of habitat
D = 1.0          # diffusion coefficient
r = 0.1          # growth rate
K = 100          # carrying capacity

Nx = 201
dx = L / (Nx - 1)

# Stability condition
dt = 0.2 * dx**2 / D

T = 100
Nt = int(T / dt)

x = np.linspace(0, L, Nx)

# =========================
# Initial population
# =========================
N = 80 * np.exp(-(x - 50)**2 / (2 * 5**2))

# =========================
# Figure
# =========================
fig, ax = plt.subplots(figsize=(10, 5))

line, = ax.plot(x, N, linewidth=3)

ax.set_xlim(0, L)
ax.set_ylim(0, K * 1.1)

ax.set_xlabel("Position x")
ax.set_ylabel("Population density N(x,t)")

title = ax.set_title("Spatial Population Growth")

# =========================
# Update function
# =========================
def update(frame):

    global N

    # Run several numerical steps
    for _ in range(10):

        N_xx = np.zeros_like(N)

        # Second derivative
        N_xx[1:-1] = (
            N[2:]
            - 2 * N[1:-1]
            + N[:-2]
        ) / dx**2

        # Reaction-diffusion equation
        N_new = (
            N
            + dt * D * N_xx
            + dt * r * N * (1 - N / K)
        )

        # Neumann boundary conditions
        N_new[0] = N_new[1]
        N_new[-1] = N_new[-2]

        N = N_new

    # Update graph
    line.set_ydata(N)

    title.set_text(
        f"Spatial Population Growth   t = {frame * 10 * dt:.1f}"
    )

    return line, title


# =========================
# Animation
# =========================
animation = FuncAnimation(
    fig,
    update,
    frames=Nt // 10,
    interval=30,
    blit=True
)

plt.show()