import numpy as np
from scipy.special import factorial
from functions.choose_t_r import choose_t_r
from functions.binom import binom
import matplotlib.pyplot as plt
from functions.SSA import my_gillespie

# ----------------------------
# ----------------------------
# RUN SIMULATION
# R1: D -> D + R
# R2: R -> R + P
# R3: R -> 0
# R4: P -> 0
# ----------------------------
# ----------------------------

# ****************************
# step A: define rate values, initial contidions values, tmax and nrmax
# ****************************

# define the stochiometry of the substrates and products separatelly
stoch_subst = np.array([[-1, 0, 0],[0, -1, 0], [0, -1, 0], [0, 0, -1]])
stoch_prods = np.array([[1,1,0],[0,1,1],[0,0,0],[0,0,0]])

# define the ci parameters (rates)
rates = np.array([0.01,0.1,0.0001,0.0001])

# define the initial conditions of the reactants
init = np.array([1,0,0])

# define the maximum time, tmax, and and maximum number of reactions, nrmax
tmax = 1
nrmax = 80

# ****************************
# step B: run simulation
# ****************************
results = my_gillespie(init, rates, stoch_subst, stoch_prods, tmax, nrmax)

# ****************************
# step C: plot results
# ****************************

# plot of simulation

# define vars
store_t = results[0]
store_mols = results[1]
store_r = results[2]

# plot results
fig, ax = plt.subplots()
ax.plot(store_t, store_mols[:,0], '-*', label='D')
ax.plot(store_t, store_mols[:,1], '-*', label='R')
ax.plot(store_t, store_mols[:,2], '-*', label='P')
legend = ax.legend(shadow=True)
plt.show()

# histogram of reactions chosen by algorithm
bins = np.array([-0.5,0.5,1.5])
plt.hist(store_r, bins)