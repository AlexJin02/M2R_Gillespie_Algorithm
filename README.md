# M2R Gillespie Algorithm

A **minimal, NumPy‑powered implementation of Gillespie’s Stochastic Simulation Algorithm (SSA)** for well‑mixed chemical reaction networks.  The core routine is `my_gillespie`, which advances the system one reaction at a time while recording the full trajectory.

---

## Features

* **Pure Python + NumPy/SciPy** – no C/Cython to compile.
* Vectorised propensity calculation for small/medium networks.
* Returns *times*, *species counts* and *reaction indices* in NumPy arrays — ready for plotting.
* Lightweight: only depends on `numpy` and `scipy`.

