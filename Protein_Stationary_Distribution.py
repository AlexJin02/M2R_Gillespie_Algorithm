import numpy as np
import matplotlib.pyplot as plt
from functions.my_gillespie import my_gillespie
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# model parameters
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = 1e-3, 1e-2, 1.2, 1.0
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
init = np.array([0, 0, 0, 0])

# ---------------------------------------------------------------------------
# long SSA run -> stationary samples
# ---------------------------------------------------------------------------
tmax = 20000.0     # total simulation time [s]
burnin = 5000.0      # discard initial transient
dt_sample = 1.0      # sample every second

ts, states, _ = my_gillespie(init.copy(), rates,
                             stoch_subst, stoch_prods,
                             tmax, nrmax=1_000_000)

# collect samples on regular grid after burn‑in
sample_times = np.arange(burnin, tmax + dt_sample, dt_sample)
samples_A = np.empty(sample_times.size, dtype=int)
samples_B = np.empty(sample_times.size, dtype=int)

idx = 0
for k, tgt in enumerate(sample_times):
    while idx + 1 < len(ts) and ts[idx + 1] <= tgt:
        idx += 1
    samples_A[k] = states[idx, 0]
    samples_B[k] = states[idx, 1]

# ---------------------------------------------------------------------------
# 2‑D empirical PMF  φ(n,m)
# ---------------------------------------------------------------------------
max_A = samples_A.max()
max_B = samples_B.max()
bins_A = np.arange(max_A + 2) - 0.5      # bin edges centred on integers
bins_B = np.arange(max_B + 2) - 0.5
H, _, _ = np.histogram2d(samples_A, samples_B,
                         bins=[bins_A, bins_B], density=True)  # probability

# ---------------------------------------------------------------------------
# Figure 1 : heat‑map of φ(n,m)
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 4.8))
im = ax1.imshow(H.T, origin='lower',
                extent=[-0.5, max_A + 0.5, -0.5, max_B + 0.5],
                aspect='auto',
                interpolation='nearest')
cb = fig1.colorbar(im, ax=ax1)
ax1.set_xlabel('number of A molecules')
ax1.set_ylabel('number of B molecules')
ax1.set_title('Stationary joint distribution φ(n, m)')
cb.set_label('probability')
plt.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2, 3 : marginal stationary distributions of A and B
# ---------------------------------------------------------------------------
pmf_A = np.bincount(samples_A, minlength=max_A+1) / samples_A.size
pmf_B = np.bincount(samples_B, minlength=max_B+1) / samples_B.size

# figure for A
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(np.arange(max_A + 1), pmf_A, width=0.8, color='tab:blue')
ax2.set_xlabel('number of A molecules')
ax2.set_ylabel('stationary probability')
ax2.set_title('Stationary marginal distribution of A')
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax2.grid(ls=':', axis='y', alpha=0.6)
plt.tight_layout()

# figure for B
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.bar(np.arange(max_B + 1), pmf_B, width=0.8, color='tab:red')
ax3.set_xlabel('number of B molecules')
ax3.set_ylabel('stationary probability')
ax3.set_title('Stationary marginal distribution of B')
ax3.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax3.grid(ls=':', axis='y', alpha=0.6)
plt.tight_layout()
plt.show()
