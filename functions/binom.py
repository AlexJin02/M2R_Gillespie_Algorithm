def binom(n, m):
    """
    Calculate the binomial coefficient C(n, m) = n! / (m! * (n - m)!).

    Parameters:
    n (int): Total number of items.
    m (int): Number of items to choose.
    Returns:
    int: The binomial coefficient C(n, m).
    """
    b = [0]*(n+1)
    b[0] = 1
    for i in range(1, n+1):
        b[i] = 1
        j = i-1
        while j > 0:
            b[j] += b[j-1]
            j -= 1
    return b[m]
