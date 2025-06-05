import numpy as np
import matplotlib.pyplot as plt
from functions.my_gillespie import my_gillespie
from matplotlib.ticker import MaxNLocator

# A + A -> ∅, A + B -> ∅, ∅ -> A, ∅ -> B
k1 = 1e-3
k2 = 1e-2
k3 = 1.2
k4 = 1.0
rates = np.array([k1, k2, k3, k4])
stoch_subst = np.array([
    [-2, 0],
    [-1, -1],
    [0, 0],
    [0, 0],
])
stoch_prods = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
])
tmax, nrmax = 600.0, 200000
init = np.array([0, 0])

ts, state, rxn_id = my_gillespie(
    init.copy(), rates, stoch_subst, stoch_prods, tmax, nrmax)

time_grid = np.arange(0, tmax + 1)
n_rep = 5
traj_grid = np.zeros((n_rep, time_grid.size), dtype=int)

# ---- simulate -----------------------------------------------------
np.random.seed(2025)
for rep in range(n_rep):
    ts, states, _ = my_gillespie(init.copy(), rates,
                                 stoch_subst, stoch_prods,
                                 tmax, nrmax)
    idx = 0
    for k, tg in enumerate(time_grid):
        while idx + 1 < len(ts) and ts[idx + 1] <= tg:
            idx += 1
        traj_grid[rep, k] = states[idx, 0]    # A molecules only

# ---------------- deterministic mean vs time -------------------------------
dt = 0.1
n_steps = int(tmax / dt) + 1
A_det = np.zeros(n_steps)
B_det = np.zeros(n_steps)
time_det = np.linspace(0, tmax, n_steps)

# initial conditions already 0
for i in range(1, n_steps):
    A = A_det[i-1]
    B = B_det[i-1]
    dA = k3 - 2*k1*A**2 - k2*A*B
    dB = k4 - k2*A*B
    A_det[i] = A + dA * dt
    B_det[i] = B + dB * dt

# ---- plot ---------------------------------------------------------
fig, ax = plt.subplots()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

for rep in range(n_rep):
    ax.step(time_grid, traj_grid[rep], where='post',
            label=f'run {rep + 1}')

ax.plot(time_det, A_det, color='black', linestyle='--', lw=2, label='Deterministic mean $\\langle B(t) \\rangle$')
ax.grid(ls=':', lw=0.6)
ax.set_xlabel('time [s]')
ax.set_ylabel('A molecules')
ax.set_title('Five SSA runs of A/B production_annihilation model')
ax.legend()
plt.tight_layout()
plt.show()
