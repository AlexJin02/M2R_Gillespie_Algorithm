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
tmax = 20000.0
burnin = 5000.0
dt_sample = 1.0
nrmax = 1_000_000
init = np.array([0, 0])

# single SSA run (quick preview)
ts, states, _ = my_gillespie(init.copy(), rates,
                             stoch_subst, stoch_prods,
                             tmax, nrmax)

# collect samples on regular grid after burn‑in
sample_times = np.arange(burnin, tmax + dt_sample, dt_sample)
RNA_samples = np.empty(sample_times.size, dtype=int)
Protein_samples = np.empty(sample_times.size, dtype=int)

idx = 0
for k, tgt in enumerate(sample_times):
    while idx + 1 < len(ts) and ts[idx + 1] <= tgt:
        idx += 1
    RNA_samples[k] = states[idx, 0]
    Protein_samples[k] = states[idx, 1]

# Empirical 2‑D stationary distribution  φ(n_RNA, n_Protein)
max_R = RNA_samples.max()
max_P = Protein_samples.max()

bins_R = np.arange(max_R + 2) - 0.5  # centre bins on integers
bins_P = np.arange(max_P + 2) - 0.5

H, _, _ = np.histogram2d(RNA_samples, Protein_samples,
                         bins=[bins_R, bins_P], density=True)

# Figure 1 : heat‑map of joint stationary distribution
fig1, ax1 = plt.subplots(figsize=(6.4, 5.6))
im = ax1.imshow(H.T, origin='lower',
                extent=[-0.5, max_R + 0.5, -0.5, max_P + 0.5],
                aspect='auto', interpolation='nearest')
cb = fig1.colorbar(im, ax=ax1)
ax1.set_xlabel('RNA molecules')
ax1.set_ylabel('Protein molecules')
ax1.set_title('Stationary joint distribution φ(n_RNA, n_Protein)')
cb.set_label('probability')
plt.tight_layout()

# Marginal stationary distributions of RNA and Protein
pmf_RNA = np.bincount(RNA_samples, minlength=max_R + 1) / RNA_samples.size
pmf_Protein = np.bincount(Protein_samples, minlength=max_P + 1) / Protein_samples.size

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(np.arange(max_R + 1), pmf_RNA, width=0.8, color='tab:blue')
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax2.set_xlabel('RNA molecules')
ax2.set_ylabel('stationary probability')
ax2.set_title('Stationary marginal distribution of RNA')
ax2.grid(ls=':', axis='y', alpha=0.6)
plt.tight_layout()

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.bar(np.arange(max_P + 1), pmf_Protein, width=0.8, color='tab:orange')
ax3.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax3.set_xlabel('Protein molecules')
ax3.set_ylabel('stationary probability')
ax3.set_title('Stationary marginal distribution of Protein')
ax3.grid(ls=':', axis='y', alpha=0.6)
plt.tight_layout()

plt.show()
