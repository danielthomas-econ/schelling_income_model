from .common import *
import numpy as np
from numba import njit, jit, prange

"------------------------------------------- cobb-douglas utility function ------------------------------------------"
# using the proportions argument to prevent redoing the math, make sure to update the proportions array appropriately
@njit(parallel = True, cache = True)
def get_utilities(agents, proportions, current_rents):
    # find utility agent i gets from moving to neighborhood k
    n = agents.size
    utilities = np.zeros((n, 100), dtype = np.float32)

    for i in prange(n):
        # skip agents that are already happy, should be a good performance boost
        if agents["happy"][i]:
            continue

        ib = agents["income_bracket"][i]
        income = agents["income"][i]
        θ = agents["theta"][i]

        row_max = 0.0
        for k in range(100):
            rent = current_rents[k]
            c = (income-rent)/income # what % of income is left after moving into the new prospective neighborhood?
            if c <= 0.0:
                c = 0.0
            q = proportions[k, ib] # quality of the new neighborhood
            val = (q ** θ) * (c ** (1-θ)) # utility fn
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
              β = BETA, # base fraction of income agent is wtp
              λ = LAMBDA, # marginal WTP for 1 unit of social utility U
              δ = DELTA): # max cap on affordability, so that bids dont take up entire agent income
    
    n_agents = agents.size
    happy = agents["happy"]
    incomes = agents["income"]
    bids = np.zeros(n_agents, dtype = np.float64)
    # which neighborhood the agents chooses to bid for
    # -1 => they're not bidding this round
    neighborhood_chosen = np.full(n_agents, -1, dtype = np.int64)   

    for i in range(n_agents):
        # checking if non-finite values might be screwing something up [TESTING PURPOSES ONLY, DELETE LATER]
        if not np.isfinite(incomes[i]):
            print(f"Non-finite income at agent {i}: {incomes[i]}")
        if not np.all(np.isfinite(utilities[i,:])):
            print(f"Non-finite utilities at agent {i}: {utilities[i,:]}")
        # agents bid if they're either not happy or not a tenant
        need_to_bid = not(happy[i])
        
        if need_to_bid:
            utility_bids = (β + λ * utilities[i,:]) * incomes[i] # utilities of all neighborhoods for agent i
            max_bids = δ * incomes[i]
            # a vector of all the potential bids the agent would make for all 100 neighborhoods
            final_bids = np.minimum(utility_bids, max_bids)

            # i think we're having some errors where nan or inf values are causing issues where the len(candidates) = 0, making random.randint bug out
            # this should weed that out since we make sure that -np.inf can never be equal to max_val
            for value in range(final_bids.shape[0]):
                if not np.isfinite(final_bids[value]):
                    final_bids[value] = -np.inf
                    
            max_val = np.max(final_bids)
            if max_val == -np.inf:
                bids[i] = 0.0
                neighborhood_chosen[i] = -1
            else:
                # tiebreaker logic since ties could be common when an agent hits his max bid cap on multiple neighborhoods
                candidates = np.where(final_bids == max_val)[0]
                k = candidates[np.random.randint(len(candidates))] # hopefully now len(candidates) will never be zero
                # set the agent's bid
                bids[i] = final_bids[k]
                # set the agent's chosen neighborhood to move them there if the bid is succcesful
                neighborhood_chosen[i] = k

    return bids, neighborhood_chosen