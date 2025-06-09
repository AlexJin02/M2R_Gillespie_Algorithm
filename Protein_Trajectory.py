import numpy as np
import matplotlib.pyplot as plt
from functions.my_gillespie import my_gillespie
from matplotlib.ticker import MaxNLocator

# A + A -> C, A + B -> D, ∅ -> A, ∅ -> B
k1 = 1e-3
k2 = 1e-2
k3 = 1.2
k4 = 1.0
rates = np.array([k1, k2, k3, k4])
stoch_subst = np.array([
    [-2, 0, 0, 0],
    [-1, -1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])
stoch_prods = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])
tmax, nrmax = 600.0, 200000
init = np.array([0, 0, 0, 0])

ts, state, rxn_id = my_gillespie(
    init.copy(), rates, stoch_subst, stoch_prods, tmax, nrmax)

time_grid = np.arange(0, tmax + 1)
n_rep = 5
traj_grid_A = np.zeros((n_rep, time_grid.size), dtype=int)
traj_grid_B = np.zeros((n_rep, time_grid.size), dtype=int)
traj_grid_C = np.zeros((n_rep, time_grid.size), dtype=int)
traj_grid_D = np.zeros((n_rep, time_grid.size), dtype=int)
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
        traj_grid_A[rep, k] = states[idx, 0]    # A molecules only
        traj_grid_B[rep, k] = states[idx, 1]    # B molecules only
        traj_grid_C[rep, k] = states[idx, 2]    # C molecules only
        traj_grid_D[rep, k] = states[idx, 3]    # D molecules only
# ---------------- deterministic mean vs time -------------------------------
dt = 0.1
n_steps = int(tmax / dt) + 1
A_det = np.zeros(n_steps)
B_det = np.zeros(n_steps)
C_det = np.zeros(n_steps)
D_det = np.zeros(n_steps)
time_det = np.linspace(0, tmax, n_steps)

# initial conditions already 0
A_det[0] = init[0]
B_det[0] = init[1]
C_det[0] = init[2]
D_det[0] = init[3]
for i in range(1, n_steps):
    A = A_det[i-1]
    B = B_det[i-1]
    C = C_det[i-1]
    D = D_det[i-1]
    dA = k3 - 2*k1*A**2 - k2*A*B
    dB = k4 - k2*A*B
    dC = k1*A**2
    dD = k2*A*B
    A_det[i] = A + dA * dt
    B_det[i] = B + dB * dt
    C_det[i] = C + dC * dt
    D_det[i] = D + dD * dt

# ---- plot ---------------------------------------------------------
fig1, ax1 = plt.subplots()
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

for rep in range(n_rep):
    ax1.step(time_grid, traj_grid_A[rep], where='post',
            label=f'run {rep + 1}')

# mean of all realisations, computed once
traj_grid_A_mean = traj_grid_A.mean(axis=0)
ax1.plot(time_grid, traj_grid_A_mean,
         color='black', lw=2, label='SSA mean ⟨A(t)⟩')
ax1.plot(time_det, A_det, color='black', linestyle='--', lw=2, label='Deterministic mean $\\langle A(t) \\rangle$')
ax1.grid(ls=':', lw=0.6)
ax1.set_xlabel('time [s]')
ax1.set_ylabel('A molecules')
ax1.set_title('Five SSA runs of A production_annihilation model')
ax1.legend()
plt.tight_layout()
plt.show()


fig2, ax2 = plt.subplots()
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
for rep in range(n_rep):
    ax2.step(time_grid, traj_grid_B[rep], where='post',
            label=f'run {rep + 1}')

# mean of all realisations, computed once
traj_grid_B_mean = traj_grid_B.mean(axis=0)
ax2.plot(time_grid, traj_grid_B_mean,
         color='black', lw=2, label='SSA mean ⟨B(t)⟩')
ax2.plot(time_det, B_det, color='black', linestyle='--', lw=2, label='Deterministic mean $\\langle B(t) \\rangle$')
ax2.grid(ls=':', lw=0.6)
ax2.set_xlabel('time [s]')
ax2.set_ylabel('B molecules')
ax2.set_title('Five SSA runs of B production_annihilation model')
ax2.legend()
plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots()
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
for rep in range(n_rep):
    ax3.step(time_grid, traj_grid_C[rep], where='post',
            label=f'run {rep + 1}')

# mean of all realisations, computed once
traj_grid_C_mean = traj_grid_C.mean(axis=0)
ax3.plot(time_grid, traj_grid_C_mean,
         color='black', lw=2, label='SSA mean ⟨C(t)⟩')
ax3.plot(time_det, C_det, color='black', linestyle='--', lw=2, label='Deterministic mean $\\langle C(t) \\rangle$')
ax3.grid(ls=':', lw=0.6)
ax3.set_ylabel('C molecules')
ax3.set_title('Five SSA runs of C production_annihilation model')
ax3.legend()
plt.tight_layout()
plt.show()

fig4, ax4 = plt.subplots()
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
for rep in range(n_rep):
    ax4.step(time_grid, traj_grid_D[rep], where='post',
            label=f'run {rep + 1}')

# mean of all realisations, computed once
traj_grid_D_mean = traj_grid_D.mean(axis=0)
ax4.plot(time_grid, traj_grid_D_mean,
         color='black', lw=2, label='SSA mean ⟨D(t)⟩')
ax4.plot(time_det, D_det, color='black', linestyle='--', lw=2, label='Deterministic mean $\\langle D(t) \\rangle$')
ax4.grid(ls=':', lw=0.6)
ax4.set_xlabel('time [s]')
ax4.set_ylabel('D molecules')
ax4.set_title('Five SSA runs of D production_annihilation model')
ax4.legend()
plt.tight_layout()
plt.show()
