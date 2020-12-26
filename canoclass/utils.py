import numpy as np

def neighbors(arr, i, j, d):
    # Use min and max to account for upper x and y neighbors
    # ranges of cells within the 'd' from the edges
    n = arr[max(i-d,0):min(i+d + 1,arr.shape[0]),
            max(j-d,0):min(j+d + 1,arr.shape[1])].flatten()
    return n
