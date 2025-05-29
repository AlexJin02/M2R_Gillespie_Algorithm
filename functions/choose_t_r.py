import numpy as np


def choose_t_r(a0, a):
    """
    Choose the time step and reaction based on the Gillespie algorithm. 

    Parameters:
    a0 (float): Total propensity.
    a (numpy.ndarray): Array of individual propensities for each reaction.

    Returns:
    T: Time step to the next reaction.
    next_r: Index of the chosen reaction.
    """
    r1, r2 = np.random.random(2)    # generate two random numbers
    # find the increment tau in time as (1/a0)*ln(1/r1)
    tau = (1 / a0) * np.log(1 / r1)

    S = sum(a)  # sum of all propensities
    mu = 0  # index for the reaction
    N = r2 * a0 - a[mu] # calculate the initial N value
    # while loop to find the next reaction
    while N > 0:
        mu += 1
        N -= a[mu]
    next_r = mu
    return tau, next_r
