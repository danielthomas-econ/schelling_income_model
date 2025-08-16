from .agents import *
from .houses import *
import numpy as np
from numba import njit, jit, prange

"------------------------------------------- cobb-douglas utility function ------------------------------------------"
# using a call to get_proportions to prevent redoing the math, make sure to update the proportions array appropriately
@njit(parallel = True, cache = True)
def get_utilities(agents, proportions):
    # find utility agent i gets from moving to neighborhood k
    n = agents.size
    utilities = np.zeros((n, 100), dtype = np.float32)

    for i in prange(n):
        ib = agents["income_bracket"][i]
        j = agents["neighborhood"][i]
        income = agents["income"][i]
        rent = agents["rent_paid"][i]
        θ = agents["theta"][i]
        # ------------ disposable income left (should this be rent after moving to k? check the logic here later) ------------ #
        c = income - rent
        c_term = c ** (1-θ) # avoids recomputation for each neighborhood

        row_max = 0.0
        for k in range(100):
            q_diff = np.maximum(0, proportions[k, ib] - proportions[j, ib])
            val = (q_diff ** θ) * c_term
            utilities[i, k] = val
            if val > row_max:
                row_max = val
    
        # normalize utility scores on a per agent basis
        # => 1 = agent's most preferred move, everything else relative to 1
        if row_max > 0.0:
            for k in range(100):
                utilities[i, k] = utilities[i, k]/row_max
    return utilities 

"--------------------------------------------------- bidding logic --------------------------------------------------"
@jit(cache = True)
def place_bid(neighborhood, start_brackets, end_brackets, incomes, tenant, rents,
              β = 0.3, # base fraction of income agent is wtp
              λ = 0.2, # marginal WTP for 1 unit of social utility U
              δ = 0.6 # max cap on affordability, so that bids dont take up entire agent income
              ):
    bids = np.zeros_like(incomes)
    happy = check_happiness(neighborhood)
    need_to_bid = ~(happy & tenant) # returns True if the agent is not both happy and a tenant, True -> must bid

    utility_bids = β*incomes + λ*get_utilities(start_brackets, end_brackets, incomes, rents)
    max_bids = δ*incomes

    final_bids = np.minimum(utility_bids, max_bids)
    # only those who need to bid have their bids updated through boolean masking
    # bids[i] remains zero if agent i is a happy tenant who wont bid
    bids[need_to_bid] = final_bids
    return bids