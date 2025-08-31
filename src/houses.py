import numpy as np
from numba import njit, jit, prange

"--------------------------------- create a structured array with all info of houses --------------------------------"
def initialize_houses(agents):
    n_agents = agents.shape[0]
    locations = np.random.uniform(0, 10, size=(n_agents, 2))
    # floor the coords and calculate neighborhood numbers
    floored = np.floor(locations).astype(int)
    x_coords = floored[:, 0]
    y_coords = floored[:, 1]
    # we have 100 1x1 neighborhoods in a 10x10 grid
    # start from 0-9 on the bottom row, all the way to 90-99 on the top row
    # so every cell number when labelled like this ends with the floor of the x coord and begins with the floor of the y coord
    neighborhood = y_coords * 10 + x_coords   

    houses_datatype = np.dtype([
        ("id", np.int32),
        ("tenant", np.int32), # which agent lives here?
        ("neighborhood", np.uint8),
        ("value", np.float64), # how much the house is actually worth, not the same as rent_paid by agent
    ])

    houses = np.zeros(n_agents, houses_datatype) 
    houses["id"] = np.arange(n_agents)
    houses["tenant"] = np.full(n_agents, -1)
    houses["neighborhood"] = neighborhood
    # TESTING THE VALUE VARIABLE TO CHECK HOMELESSNESS
    houses["value"] = np.full(n_agents, 1_00_000) # assuming median 2bhk price is 30k/m, so 3.6L pa
    return houses

"--------------------------------------- populate the houses column in agents ---------------------------------------"
def agent_house_mapping(agents, houses):
    n_agents = agents.shape[0]
    n_houses = houses.shape[0]
    # we set everyone to -1 and then map their houses later
    # otherwise we run into problems with negative indexing for homeless people
    agents["house"][:] = -1
    for h in range(n_houses):
        t = houses["tenant"][h]
        if t != -1: # we dont assign homeless tenants houses
            agents["house"][t] = h
    return
"--------------------------------- check if an agent can no longer afford their home --------------------------------"
"""We define the agent's max WTP for rent in neighborhood $j$ as
$B_{i}^{\text{stay}}= \text{min}(\beta Y_{i} + \lambda Q_i, \delta Y_i)$
The core logic is identical to bidding, but we had to replace $U$ since that compares two neighborhoods
We replace it with $Q_{i} = r_{k}$ percent of agents in neighborhood $k$ with >= income to eliminate the comparative nature of $U$."""
@njit(parallel = True, cache = True)
def check_priced_out(agents, houses, proportions, β=0.3, λ=0.2, δ=0.4): # removing proportions as an argument causes numba issues
    rents = houses["value"] # base it off the current house value, not how much the agent pays for rent
    n_agents = agents.shape[0]
    priced_out_mask = np.zeros(n_agents, dtype=np.bool_)

    for i in prange(n_agents):
        nb = agents["neighborhood"][i]
        if nb == -1: # homeless agents cant be priced out
            continue

        income = agents["income"][i]
        bracket = agents["income_bracket"][i]
        Q_i = proportions[nb, bracket]
        B_stay = min(β*income + λ*Q_i, δ*income) # priced out logic
        if rents[i] > B_stay:
            priced_out_mask[i] = True
    
    return priced_out_mask

"---------------------------------- evict the poor dudes who are now priced out :( ----------------------------------"
def evict_priced_out(agents, houses, priced_out_mask):
    n_agents = agents.shape[0]
    for i in range(n_agents):
        h = agents["house"][i] # corresponds to house id they live in
        if priced_out_mask[i]:
            if h != -1: # agent still lives somewhere
                houses["tenant"][h] = -1 # evict them
                agents["house"][i] = -1
                agents["neighborhood"][i] = -1
    return
"---------------------------------------- decide which agent gets which house ---------------------------------------"
@njit(cache=True)
def allocate_houses(agents, houses, bids, neighborhood_chosen, β=0.3, λ=0.2, δ=0.4):
    n_agents = agents.shape[0]
    n_houses = houses.shape[0]
    n_neighborhoods = np.max(houses["neighborhood"])+1
    cutoff_bids = np.zeros(n_neighborhoods)
    vacant_mask = houses["tenant"] == -1

    # run an auction in each neighborhood
    for n in range(n_neighborhoods):
        # list of all vacant houses in our current neighborhood
        # outputs some weird format where 1st item in the list is what we  want
        vacancies = np.where(vacant_mask & (houses["neighborhood"]==n))[0]
        k = vacancies.shape[0]
        if k == 0:
            continue

        # list of all bidders in the neighborhood
        bidders = np.where(neighborhood_chosen == n)[0]
        if bidders.shape[0] == 0:
            continue
        
        # sort bids in descending order
        order = np.argsort(-bids[bidders]) # -> argsort gives indices
        sorted_bidders = bidders[order] # use those indices to sort here
        winners = sorted_bidders[:k] # -> anyone beyond k (num vacancies) automatically loses
        if winners.shape[0] > 0:
            cutoff_bids[n] = bids[winners[-1]] # gives us the value of the lowest successful bid for each neighborhood

        # give winners their vacant homes
        for w, v in zip(winners, vacancies):
            houses["tenant"][v] = w
            houses["value"][v] = bids[w]
            agents["house"][w] = v
            agents["rent_paid"][w] = bids[w]
            agents["neighborhood"][w] = n

    return agents, houses, cutoff_bids

"-------------------------------------- update the house prices based on demand -------------------------------------"
def update_prices(houses, bids, neighborhood_chosen, cutoff_bids,
                  α = 0.3): # sensitivity of price movements
    n_neighborhoods = np.max(houses["neighborhood"]) + 1
    
    for n in range(n_neighborhoods):
        mask = (neighborhood_chosen == n)
        bids_here = bids[mask] # the bids for this neighborhood
        winners = bids_here[bids_here >= cutoff_bids[n]] # those who bid more than the cutoff won
        
        if len(winners) > 0: 
            B = np.mean(winners)
            P = houses["value"][houses["neighborhood"]==n] # price of all houses in neighborhood n
            σ = 0.02*np.mean(P) # array of std devs
            ε = np.random.normal(0,σ, P.shape) # array of noise
            new_price = P + α*(B-P) + ε
            new_price = np.maximum(new_price,0) # avoids negative prices
            houses["value"][houses["neighborhood"]==n] = new_price # update all prices in the neighborhood

    return houses