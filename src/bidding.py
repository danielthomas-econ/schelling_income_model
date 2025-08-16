from agents import *
from houses import *
import numpy as np
from numba import njit, jit

"------------------------------------------- cobb-douglas utility function ------------------------------------------"
# USE A CALL TO get_freq_and_total TO PREVENT DOING ALL THE MATH AGAIN HERE
@njit
def utility(θ, # gen a ~uniform(0.6,0.8) array to get some heterogenity, using it as an argument so i can njit the function
            brackets_start, brackets_end, income, rent):
    
    freq_start = np.bincount(brackets_start, minlength=12)
    freq_end = np.bincount(brackets_end, minlength=12)
    n_start = brackets_start.size
    n_end = brackets_end.size
    # same logic as the check_happiness function
    cumsum_start = n_start - np.cumsum(freq_start) + freq_start
    cumsum_end = n_end - np.cumsum(freq_end) + freq_end

    # percent of popln with >= income bracket b in the two neighborhoods we're comparing
    quality_start = (cumsum_start / n_start) * 100
    quality_end = (cumsum_end / n_end) * 100

    # q_start[b] = % with >= than income bracket b in starting neighborhood
    # vectorized over all the brackets in starting neighborhood
    q_start = quality_start[brackets_start]
    q_end = quality_end[brackets_start]

    q_diff = np.maximum(0, q_end - q_start)
    c = income - rent
    # utils[i] is utility of agent i gets for moving from neighborhood j to k
    utils = q_diff ** {θ} * c ** (1-θ)
    return utils

"--------------------------------------------------- bidding logic --------------------------------------------------"
@jit
def place_bid(neighborhood, start_brackets, end_brackets, incomes, tenant, rents,
              β = 0.3, # base fraction of income agent is wtp
              λ = 0.2, # marginal WTP for 1 unit of social utility U
              δ = 0.6 # max cap on affordability, so that bids dont take up entire agent income
              ):
    bids = np.zeros_like(incomes)
    happy = check_happiness(neighborhood)
    need_to_bid = ~(happy & tenant) # returns True if the agent is not both happy and a tenant, True -> must bid

    utility_bids = β*incomes + λ*utility(start_brackets, end_brackets, incomes, rents)
    max_bids = δ*incomes

    final_bids = np.minimum(utility_bids, max_bids)
    # only those who need to bid have their bids updated through boolean masking
    # bids[i] remains zero if agent i is a happy tenant who wont bid
    bids[need_to_bid] = final_bids
    return bids