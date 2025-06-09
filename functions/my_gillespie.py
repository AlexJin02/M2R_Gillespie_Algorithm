import numpy as np
from scipy.special import binom, factorial
from .choose_t_r import choose_t_r
from .binom import binom


def my_gillespie(init, rates, stoch_subst, stoch_prods, tmax, nrmax):
    """
    Gillespie algorithm for simulating a stochastic chemical reaction system.

    Parameters
    ----------
    init: Initial number of molecules of each species.
    rates: Reaction rates for each reaction.
    stoch_subst: Stoichiometry of reactants for each reaction (rows are reactions, columns are species).
    stoch_prods: Stoichiometry of products for each reaction (rows are reactions, columns are species).
    tmax: Maximum time for the simulation.
    nrmax: Maximum number of reactions to simulate.
    Returns
    -------
    store_t: Times at which reactions occur.
    store_mols: Number of molecules of each species at each time point.
    store_r: Indices of reactions that occurred at each time point.
    """
    # step 0: input rate values, initial contidions values and initialise time and reactions counter
    stoch = stoch_subst + stoch_prods   # stoichiometry (variable stoch) of the system
    num_rxn = stoch.shape[0]    # number of reactions
    num_spec = stoch.shape[1]   # number of species
    current_species = init.copy()
    current_t = 0
    react_count = 0
    t_count = 0
    store_t = np.zeros((2 * nrmax, 1))  # store times
    store_mols = np.zeros((2 * nrmax, num_spec))    # store molecules
    store_r = np.zeros((2 * nrmax, 1))  # store reactions
    store_t[0] = current_t
    store_mols[0, :] = current_species

    # main while loop
    while react_count < nrmax and current_t < tmax:

        # step 1: calculate ai and a0
        a = np.ones((num_rxn, 1))   # initialise a to ones array of dimensions num_rxn,1
        for i in range(num_rxn):
            hi = 1
            for j in range(len(init)):
                if stoch_subst[i, j] == 0:    # species j not consumed in reaction i
                    continue
                if current_species[j] <= abs(stoch_subst[i, j]):    # Not enough molecules
                    hi = 0
                    break
                else:
                    # hi is calculated as the product of the binomial coefficient and factorial
                    hi *= binom(current_species[j], np.abs(stoch_subst[i, j])) * factorial(np.abs(stoch_subst[i, j]))
            a[i] = hi * rates[i]  # save the current value of ai as hi*ci
        a0 = sum(a)  # save a0 as the sum of all a's

    # step 2: choose next t and r
        T, next_r = choose_t_r(a0, a)  # run choose_t_r to get the change in time and the next reaction

    # step 3: update and store system
        current_t += T  # update current_t by adding T
        current_species += np.transpose(stoch[next_r, :])  # update current_species
        react_count += 1  # update the reaction counter
        t_count += 1  # update the time counter
        store_t[t_count] = current_t  # store current time
        store_mols[t_count, :] = current_species  # store current numbers of molecules
        store_r[t_count] = next_r  # store the reaction id

    # get rid of empty entries in store_t, store_mols, store_r
    store_t = store_t[:t_count + 1]
    store_mols = store_mols[:t_count + 1, :]
    store_r = store_r[:t_count + 1]

    # return the results
    return store_t, store_mols, store_r
