import numpy as np
from numba import njit


"------------------------------ list of all houses (indices matches neighborhood_list) ------------------------------"
@njit
def initialize_houses(neighborhood):
    houses = [np.full_like(array, 3_60_000) for array in neighborhood] # assuming median 2bhk price is 30k/m, so 3.6L pa
    vacancies = [np.zeros_like(array) for array in neighborhood] # we'll use this to put 1 for taken, 0 for vacant
    return houses, vacancies



