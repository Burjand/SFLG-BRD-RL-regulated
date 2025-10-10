import numpy as np

def pack(a, n, distribution):

    """
    Create n boxes of values taking into consideration the data distribution and then change the original vector to accomodate those values
    """

    # Check if values are appropiate
    a = np.asarray(a)
    if a.size == 0:
        return np.array([], dtype=int)
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer")

    if distribution == 'uniform':

        minimum = float(a.min())
        maximum = float(a.max())
        if maximum == minimum:
            # Degenerate: all equal -> assign all to bin 0
            return np.zeros_like(a, dtype=int)
        # Create n+1 edges equally spaced from min to max
        edges = np.linspace(minimum, maximum, n + 1)
        # Searchsorted with side='right' and subtract 1 to place values in [edge_i, edge_{i+1})
        idx = np.searchsorted(edges, a, side='right') - 1
        packed_a = np.clip(idx, 0, n - 1)

    elif distribution == 'normal':

        pass

    else:
        raise Exception("Unknown distribution")
    
    return packed_a
    

a = [1,2,3,4,5,6,7,8,9,10]

print(pack(a,5,'uniform'))