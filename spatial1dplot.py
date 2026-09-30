import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
L = 100.0       # length of habitat
D = 1.0         # diffusion coefficient
r = 0.1         # growth rate
K = 100.0       # carrying capacity

T = 100.0       # simulation time

# =========================
# Numerical grid
# =========================
Nx = 201
dx = L / (Nx - 1)

# Stability condition for explicit scheme
dt = 0.2 * dx**2 / D

Nt = int(T / dt)

x = np.linspace(0, L, Nx)

# =========================
# Initial population
# =========================
N = 100 * np.exp(-(x - 50)**2 / (2 * 5**2))

# =========================
# Plot initial condition
# =========================
plt.figure(figsize=(9, 5))
plt.plot(x, N)
plt.xlabel("Position x")
plt.ylabel("Population density N(x,0)")
plt.title("Initial population distribution")
plt.grid()
plt.show()

# =========================
# Time evolution
# =========================
snapshots = []
times = []

save_every = max(1, Nt // 5)

for n in range(Nt):

    # Second derivative
    N_xx = np.zeros_like(N)

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

    # Neumann boundary condition
    N_new[0] = N_new[1]
    N_new[-1] = N_new[-2]

    N = N_new

    if n % save_every == 0:
        snapshots.append(N.copy())
        times.append(n * dt)

# =========================
# Plot snapshots
# =========================
plt.figure(figsize=(10, 6))

for N_snapshot, t in zip(snapshots, times):
    plt.plot(x, N_snapshot, label=f"t = {t:.1f}")

plt.xlabel("Position x")
plt.ylabel("Population density")
plt.title("Spatial population growth")
plt.legend()
plt.grid()
plt.show()


# =========================
# Heatmap
# =========================

# Store full solution
N = 100 * np.exp(-(x - 50)**2 / (2 * 5**2))

solution = []
time_values = []

for n in range(Nt):

    N_xx = np.zeros_like(N)

    N_xx[1:-1] = (
        N[2:]
        - 2 * N[1:-1]
        + N[:-2]
    ) / dx**2

    N_new = (
        N
        + dt * D * N_xx
        + dt * r * N * (1 - N / K)
    )

    # Neumann boundary condition
    N_new[0] = N_new[1]
    N_new[-1] = N_new[-2]

    N = N_new

    if n % 10 == 0:
        solution.append(N.copy())
        time_values.append(n * dt)

solution = np.array(solution)

plt.figure(figsize=(10, 6))

plt.imshow(
    solution,
    aspect="auto",
    extent=[0, L, 0, T],
    origin="lower"
)

plt.xlabel("Position x")
plt.ylabel("Time t")
plt.title("Spatial population dynamics")

plt.colorbar(label="Population density")

plt.show()