import numpy as np
from scipy.special import factorial
from functions.choose_t_r import choose_t_r
from functions.binom import binom
import matplotlib.pyplot as plt
from functions.SSA import my_gillespie


# parameters
c = 1.0
x0 = 100
tmax = 7.0
nrmax = 20000
init = np.array([x0])
rates = np.array([c])
stoch_subst = np.array([[-1]])  # only one substrate
stoch_prods = np.array([[0]])   # the product is empty

# deterministic mean & SD
t_dense = np.linspace(0, tmax, 400)
mean_det = x0 * np.exp(-c * t_dense)
sd_det = np.sqrt(x0 * np.exp(-c * t_dense) * (1 - np.exp(-c * t_dense)))

# run my_gillespie algorithm to generate SSA results
np.random.seed(2025)    # make sure the results is reproducable
times_list = []     # store the event times for each SSA replicate
x_list = []     # store the molecule-count trajectory of the species S for each replicate.

# run two independent SSA simulations.
for _ in range(2):
    ts, mols, r = my_gillespie(init.copy(), rates, stoch_subst, stoch_prods,
                               tmax, nrmax)
    times_list.append(ts)
    x_list.append(mols[:, 0])

# plot the SSA trajectories
fig, ax = plt.subplots()
colors = ['tab:red', 'tab:blue']
for idx in range(2):
    ax.step(times_list[idx], x_list[idx], where='post',
            color=colors[idx], linewidth=1.2, label=f'SSA trajectory {idx+1}')

# plot the deterministic RRE mean ± SD
ax.plot(t_dense, mean_det, color='steelblue', linewidth=1.0, label='Deterministic RRE mean')
ax.plot(t_dense, mean_det + sd_det, color='grey', linestyle='--', linewidth=1.0, label='Mean ± SD (CME)')
ax.plot(t_dense, mean_det - sd_det, color='grey', linestyle='--', linewidth=1.0)

ax.set_xlabel('time')
ax.set_ylabel('X(t)')
ax.set_xlim(0, tmax)
ax.set_ylim(0, x0 + 2)
ax.set_title('Simple decay  S → ∅  (c = 1,  X₀ = 100)')
ax.legend()
plt.tight_layout()
plt.show()