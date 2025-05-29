# M2R Gillespie Algorithm

A **minimal, NumPy‑powered implementation of Gillespie’s Stochastic Simulation Algorithm (SSA)** for well‑mixed chemical reaction networks.  The core routine is `my_gillespie`, which advances the system one reaction at a time while recording the full trajectory.

---

## Features

* **Pure Python + NumPy/SciPy**
* Vectorised propensity calculation.
* Returns *times*, *species counts* and *reaction indices* in NumPy arrays — ready for plotting.
