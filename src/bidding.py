from .agents import *
from .houses import *
import numpy as np
from numba import njit, jit, prange

"------------------------------------------- cobb-douglas utility function ------------------------------------------"
# using the proportions argument to prevent redoing the math, make sure to update the proportions array appropriately
@njit(parallel = True, cache = True)
def get_utilities(agents, proportions):
    # find utility agent i gets from moving to neighborhood k
    n = agents.size
    utilities = np.zeros((n, 100), dtype = np.float32)

    for i in prange(n):
        ib = agents["income_bracket"][i]
        j = agents["neighborhood"][i]
        if j == -1:
            proportions[j,ib] = 0.0 # anything is better than being homeless
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
def place_bid(agents, utilities,
              β = 0.3, # base fraction of income agent is wtp
              λ = 0.2, # marginal WTP for 1 unit of social utility U
              δ = 0.6): # max cap on affordability, so that bids dont take up entire agent income
    
    n_agents, n_neighborhoods = utilities.shape
    happy = agents["happy"]
    incomes = agents["income"]
    rent_paid = agents["rent_paid"]
    bids = np.zeros(n_agents, dtype = np.float64)
    # which neighborhood the agents chooses to bid for
    # -1 => they're not bidding this round
    neighborhood_chosen = np.full(n_agents, -1, dtype = np.int64)   

    for i in range(n_agents):
        # agents bid if they're either not happy or not a tenant
        need_to_bid = not(happy[i] and rent_paid[i]>0) # using rent_paid = 0 as a proxy for non-tenancy
        
        if need_to_bid:
            utiliity_bids = β * incomes[i] + λ * utilities[i,:] # utilities of all neighborhoods for agent i
            max_bids = δ * incomes[i]
            # a vector of all the potential bids the agent would make for all 100 neighborhoods
            final_bids = np.minimum(utiliity_bids, max_bids)
            max_val = np.max(final_bids)
            # tiebreaker logic since ties could be common when an agent hits his max bid cap on multiple neighborhoods
            candidates = np.where(final_bids == max_val)[0]
            k = candidates[np.random.randint(len(candidates))]
            # set the agent's bid
            bids[i] = final_bids[k]
            # set the agent's chosen neighborhood to move them there if the bid is succcesfful
            neighborhood_chosen[i] = k

    return bids, neighborhood_chosen