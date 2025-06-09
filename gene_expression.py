import numpy as np
import matplotlib.pyplot as plt
from functions.my_gillespie import my_gillespie
from matplotlib.ticker import MaxNLocator

# 1. DNA → DNA + RNA
# 2. RNA → RNA + Protein
# 3. RNA → ∅
# 4. Protein → ∅

# reaction rates
k1 = 0.05
k2 = 0.10
k3 = 0.01
k4 = 0.001
rates = np.array([k1, k2, k3, k4])
stoch_subst = np.array([
    [0, 0],
    [-1, 0],
    [-1, 0],
    [0, -1],
])
# Product stoichiometry
stoch_prods = np.array([
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
])

# simulation settings
tmax, nrmax = 1000.0, 500_000
init = np.array([0, 0])

# single SSA run (quick preview)
ts, states, _ = my_gillespie(init.copy(), rates,
                             stoch_subst, stoch_prods,
                             tmax, nrmax)

# time grid for aligning trajectories and plotting
wall_dt = 1.0                    # minutes
time_grid = np.arange(0, tmax + wall_dt, wall_dt)

# replicate ensemble
n_rep = 5
traj_grid_R = np.zeros((n_rep, time_grid.size), dtype=int)
traj_grid_P = np.zeros((n_rep, time_grid.size), dtype=int)

np.random.seed(2025)
for rep in range(n_rep):
    ts_rep, states_rep, _ = my_gillespie(init.copy(), rates,
                                         stoch_subst, stoch_prods,
                                         tmax, nrmax)
    idx = 0
    for k, tg in enumerate(time_grid):
        while idx + 1 < len(ts_rep) and ts_rep[idx + 1] <= tg:
            idx += 1
        traj_grid_R[rep, k] = states_rep[idx, 0]   # RNA molecules
        traj_grid_P[rep, k] = states_rep[idx, 1]   # Protein molecules

# deterministic ODE solution
dt = wall_dt
n_steps = int(tmax / dt) + 1
R_det = np.zeros(n_steps)
P_det = np.zeros(n_steps)
time_det = np.linspace(0, tmax, n_steps)

R_det[0] = init[0]
P_det[0] = init[1]
for i in range(1, n_steps):
    R = R_det[i-1]
    P = P_det[i-1]
    dR = k1 - k3 * R
    dP = k2 * R - k4 * P
    R_det[i] = R + dR * dt
    P_det[i] = P + dP * dt

# plotting RNA
fig1, ax1 = plt.subplots()
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
for rep in range(n_rep):
    ax1.step(time_grid, traj_grid_R[rep], where='post',
             label=f'run {rep + 1}')
traj_grid_R_mean = traj_grid_R.mean(axis=0)
ax1.plot(time_grid, traj_grid_R_mean, color='black', lw=2,
         label='SSA mean ⟨mRNA(t)⟩')
ax1.plot(time_det, R_det, color='black', linestyle='--', lw=2,
         label='Deterministic ⟨mRNA(t)⟩')
ax1.grid(ls=':', lw=0.6)
ax1.set_xlabel('time [min]')
ax1.set_ylabel('mRNA molecules')
ax1.set_title('Five SSA runs of the gene‑expression model (mRNA)')
ax1.legend()
plt.tight_layout()
plt.show()

# plotting Protein
fig2, ax2 = plt.subplots()
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
for rep in range(n_rep):
    ax2.step(time_grid, traj_grid_P[rep], where='post',
             label=f'run {rep + 1}')
traj_grid_P_mean = traj_grid_P.mean(axis=0)
ax2.plot(time_grid, traj_grid_P_mean, color='black', lw=2,
         label='SSA mean ⟨Protein(t)⟩')
ax2.plot(time_det, P_det, color='black', linestyle='--', lw=2,
         label='Deterministic ⟨Protein(t)⟩')
ax2.grid(ls=':', lw=0.6)
ax2.set_xlabel('time [min]')
ax2.set_ylabel('Protein molecules')
ax2.set_title('Five SSA runs of the gene‑expression model (Protein)')
ax2.legend()
plt.tight_layout()
plt.show()
