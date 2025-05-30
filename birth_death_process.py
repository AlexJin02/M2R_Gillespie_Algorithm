import numpy as np
from scipy.special import factorial
from functions.choose_t_r import choose_t_r
from functions.binom import binom
import matplotlib.pyplot as plt
from functions.my_gillespie import my_gillespie

# Consider the reaction A -k1-> Ø, Ø -k2-> A
# parameters
k1 = 0.1
k2 = 1.0
x0 = 0
tmax = 100.0
nrmax = 50000
init = np.array([x0])
rates = np.array([k1, k2])
stoch_subst = np.array([[-1],[0]])
stoch_prods = np.array([[0],[1]])

time_grid = np.arange(0, tmax+1)
n_rep = 5
traj_grid = np.zeros((n_rep, time_grid.size))

# Store the SSA trajectory in traj_grid
np.random.seed(2025)
for r in range(n_rep):
    ts, states, _ = my_gillespie(init.copy(), rates, stoch_subst, stoch_prods,
                              tmax, nrmax)
    idx = 0
    for k, tg in enumerate(time_grid):
        while idx + 1 < len(ts) and ts[idx+1] <= tg:
            idx += 1
        traj_grid[r, k] = states[idx]

# analytic mean M(t) = (k2/k1)(1 - exp(-k1 t))
analytic_mean = (k2/k1)*(1 - np.exp(-k1*time_grid))
analytic_std = np.sqrt(analytic_mean)

# plot
fig, ax = plt.subplots()
for r in range(n_rep):
    ax.plot(time_grid, traj_grid[r], lw=1, label=f'SSA trajectory {r+1}')
ax.plot(time_grid, analytic_mean, linewidth=1.3, label='Deterministic RRE mean')
ax.plot(time_grid, analytic_mean + analytic_std, color='grey', linestyle='--', linewidth=1.0, label='Mean ± SD (CME)')
ax.plot(time_grid, analytic_mean - analytic_std, color='grey', linestyle='--', linewidth=1.0)
ax.set_xlabel('time [s]')
ax.set_ylabel('A molecules')
ax.set_title('Five SSA runs of production-degradation system')
ax.legend()
plt.tight_layout()
plt.show()

# Stationary distribution
long_time = 100000     # seconds
ts_long, states_long, r = my_gillespie(init.copy(), rates, stoch_subst, stoch_prods, long_time, 1000000)

# sample once per second
sampled = []
idx = 0
for t in np.arange(0, long_time):
    while idx + 1 < len(ts_long) and ts_long[idx+1] <= t:
        idx += 1
    sampled.append(states_long[idx, 0])
sampled = np.array(sampled)

lam = k2/k1
n_vals = np.arange(sampled.min(), sampled.max()+1)
poisson_pmf = np.exp(-lam) * lam**n_vals / np.array([factorial(n) for n in n_vals])

# plot
fig2, ax2 = plt.subplots()
ax2.hist(sampled, bins=np.arange(n_vals.min()-0.5, n_vals.max()+1.5), density=True, alpha=0.6, label='SSA histogram')
ax2.plot(n_vals, poisson_pmf, lw=2, label='Analytic Poisson pmf')
ax2.set_xlabel('A molecules')
ax2.set_ylabel('stationary probability')
ax2.set_title('Stationary distribution after long SSA run')
ax2.legend(frameon=False)
plt.tight_layout()
plt.show()