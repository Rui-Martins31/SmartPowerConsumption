import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

#---------------------------
def plot_dimension(num_rows:int, num_col:int):
    pass

#---------------------------
def find_local_maxmin(data_list: list[float], maxmin: str = "max", smooth_area: int = 1, smooth_threshold: float = 0.01) -> (tuple[list[int], list[float]] | None):
    """
    Finds local maximums or minimums in a list.
    """

    arr: np.ndarray = np.array(data_list)
    if maxmin == "max":
        maxmin_indices = argrelextrema(arr, np.greater)[0]
    elif maxmin == "min": 
        maxmin_indices = argrelextrema(arr, np.less)[0]
    else:
        print("maxmin is not valid!")
        return None

    # Get neighbors
    neigh_indices: list[int] = []
    for idx in maxmin_indices:
        for incr in range(smooth_area, 0, -1):
            if (idx-incr >= 0):
                if (arr[idx - incr] <= arr[idx] * 1.0+smooth_threshold) and (arr[idx - incr] >= arr[idx] * 1.0-smooth_threshold):
                    neigh_indices.append(idx - incr)
        
        neigh_indices.append(idx)

        for incr in range(1, smooth_area+1, 1):
            if (idx+incr <= len(arr) - 1):
                if (arr[idx + incr] <= arr[idx] * 1.0+smooth_threshold) and (arr[idx + incr] >= arr[idx] * 1.0-smooth_threshold):
                    neigh_indices.append(idx + incr)

    # Non-repeatable
    neigh_indices: set[int] = set(neigh_indices)
    neigh_indices: list[int] = list(neigh_indices)
    
    #return arr[maxmin_indices].tolist()
    #return maxmin_indices, arr[maxmin_indices].tolist()
    return neigh_indices, arr[neigh_indices].tolist()