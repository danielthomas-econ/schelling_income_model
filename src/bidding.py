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
            current_prop = 0.0 # setting proportions[j,ib] = 0 in a prange loop is unsafe, using this variable as an intermediary for that
        else:
            current_prop = proportions[j,ib]
        income = agents["income"][i]
        rent = agents["rent_paid"][i]
        θ = agents["theta"][i]
        # ------------ disposable income left (should this be rent after moving to k? check the logic here later) ------------ #
        c = income - rent
        if c <= 0.0:
            c_term = 0.0 # no utility if rents exceed your income, thats impossible
        else:
            c_term = c ** (1-θ) # avoids recomputation for each neighborhood

        row_max = 0.0
        for k in range(100):
            q_diff = np.maximum(0, proportions[k, ib] - current_prop)
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
        # checking if non-finite values might be screwing something up [TESTING PURPOSES ONLY, DELETE LATER]
        if not np.isfinite(incomes[i]):
            print(f"Non-finite income at agent {i}: {incomes[i]}")
        if not np.all(np.isfinite(utilities[i,:])):
            print(f"Non-finite utilities at agent {i}: {utilities[i,:]}")
        # agents bid if they're either not happy or not a tenant
        need_to_bid = not(happy[i] and rent_paid[i]>0) # using rent_paid = 0 as a proxy for non-tenancy
        
        if need_to_bid:
            utility_bids = β * incomes[i] + λ * utilities[i,:] # utilities of all neighborhoods for agent i
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