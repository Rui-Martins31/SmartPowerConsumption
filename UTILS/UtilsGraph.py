import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

#---------------------------
def plot_dimension(num_rows:int, num_col:int):
    pass

#---------------------------
def find_local_maxmin(data_list: list[float], 
                      maxmin: str = "max", 
                      smooth_area: int = 1, 
                      smooth_threshold: float = 0.01,
                      moving_average: list[float] = []) -> (tuple[list[int], list[float]] | None):
    """
    Finds local maximums or minimums in a list.
    """

    arr: np.ndarray = np.array(data_list)
    if maxmin == "max":
        maxmin_indices = argrelextrema(arr, np.greater)[0]
    elif maxmin == "min": 
        # if moving_average:
        #     avg: float = average(data_list)
        #     arr: np.ndarray = np.where(arr <= avg, arr, float("inf"))
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

    ## Check if it's below the moving average
    ## Temporary
    if maxmin == "min" and moving_average != []:
        temp_list_idx: list[int] = []
        avg: float = average(moving_average)
        for idx in range(len(arr)):
            if arr[idx] <= avg:
                temp_list_idx.append(idx)
        neigh_indices: list[int] = temp_list_idx

    # Non-repeatable
    neigh_indices: set[int] = set(neigh_indices)
    neigh_indices: list[int] = list(neigh_indices)
    
    #return arr[maxmin_indices].tolist()
    #return maxmin_indices, arr[maxmin_indices].tolist()
    return neigh_indices, arr[neigh_indices].tolist()

#---------------------------
def average(value_list: list) -> float:
    return float(sum(value_list)/len(value_list))