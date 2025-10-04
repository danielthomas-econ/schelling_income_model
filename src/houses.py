from .common import *
import numpy as np
from numba import njit, jit, prange

"--------------------------------- create a structured array with all info of houses --------------------------------"
def initialize_houses(agents, starting_house_price = STARTING_HOUSE_PRICE):
    n_agents = agents.size

    houses_datatype = np.dtype([
        ("id", np.int32),
        ("tenant", np.int32), # which agent lives here?
        ("neighborhood", np.uint8),
        ("value", np.float64), # how much the house is actually worth, not the same as rent_paid by agent
    ])

    houses = np.zeros(n_agents, houses_datatype) 
    houses["id"] = np.arange(n_agents)
    houses["tenant"] = np.full(n_agents, -1)
    houses["neighborhood"] = allocate_neighborhood(agents)
    houses["value"] = np.full(n_agents, starting_house_price)
    return houses

"--------------------------------------- populate the houses column in agents ---------------------------------------"
@njit(cache = True)
def agent_house_mapping(agents, houses):
    # we check if agent's house matches house's agent
    # if they don't, the agent is homeless and we give him nonmarket housing
    for i in range(len(agents)):
        if agents["house"][i] != -1: # they have a home
            current_house = agents["house"][i]
            if houses["tenant"][current_house] != i: # mismatch between agent and houses array
                agents["house"][i] = -1
                agents["nonmarket_housing"][i] = True
                agents["rent_paid"][i] = 0.0 # you dont pay rent for nonmarket housing

    for h in range(len(houses)):
        t = houses["tenant"][h] # agent id living in home h
        if t != -1: # we dont assign homeless tenants houses
            agents["house"][t] = h # agent t lives in home h
            agents["neighborhood"][t] = houses["neighborhood"][h] # matches agent and home's neighborhoods"""
            agents["nonmarket_housing"][t] = False # they have a market home now
    return
"--------------------------------- check if an agent can no longer afford their home --------------------------------"
@njit(parallel = True, cache = True)
def check_priced_out(agents, houses, proportions, β=BETA, λ=LAMBDA, δ=DELTA):
    n_agents = agents.size
    priced_out_mask = np.zeros(n_agents, dtype=np.bool_)

    for i in prange(n_agents):
        h = agents["house"][i]
        if h == -1:
            continue # homeless agents dont have a home they can be priced out of anyways
        
        nb = agents["neighborhood"][i]
        income = agents["income"][i]
        bracket = agents["income_bracket"][i]
        Q_i = proportions[nb, bracket]
        B_stay = min((β + λ*Q_i)*income, δ*income) # priced out logic
        rent = houses["value"][h] # base it off the current house value, not how much the agent pays for rent
        if rent > B_stay:
            priced_out_mask[i] = True
    
    return priced_out_mask

"---------------------------------- evict the poor dudes who are now priced out :( ----------------------------------"
def evict_priced_out(agents, houses, priced_out_mask):
    n_agents = agents.size
    for i in range(n_agents):
        h = agents["house"][i] # corresponds to house id they live in
        if priced_out_mask[i]:
            if h != -1: # agent still lives somewhere
                houses["tenant"][h] = -1 # evict them
    # besides eviction, agent_house_mapping will take care of the rest of the bookkeeping stuff
    agent_house_mapping(agents,houses)
    return
"---------------------------------------- decide which agent gets which house ---------------------------------------"
@njit(cache=True)
def allocate_houses(agents, houses, bids, neighborhood_chosen):
    n_neighborhoods = np.max(houses["neighborhood"])+1
    cutoff_bids = np.zeros(n_neighborhoods)
    vacant_mask = houses["tenant"] == -1
    num_winners = 0 # initialize it outside the loop to prevent errors when zero bidders/vacancies are there

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
        num_winners = len(winners) # to add to the stats array
        
        if winners.shape[0] > 0:
            cutoff_bids[n] = bids[winners[-1]] # gives us the value of the lowest successful bid for each neighborhood

        # give winners their vacant homes
        for w, v in zip(winners, vacancies):
            houses["tenant"][v] = w
            agents["house"][w] = v
            agents["neighborhood"][w] = n
            agents["nonmarket_housing"][w] = False
            # all houses share the same price: the market clearing price
            houses["value"][v] = cutoff_bids[n]
            agents["rent_paid"][w] = cutoff_bids[n]

    agent_house_mapping(agents, houses) # update the mapping after the allocation is made
    return agents, houses, cutoff_bids, num_winners

"-------------------------------------- update the house prices based on demand -------------------------------------"
def update_prices(agents, houses, neighborhood_chosen, cutoff_bids,
                  decay_rate = DECAY_RATE, # fall in price if supply > demand
                  max_change = MAX_CHANGE,
                  β = BETA): # maximum % change in price in one round
    n_neighborhoods = np.max(houses["neighborhood"]) + 1
    
    for n in range(n_neighborhoods):
        mask = houses["neighborhood"] == n
        agents_mask = agents["neighborhood"] == n
        old_price = houses["value"][mask] # current prices
        vacancies = np.sum(mask & (houses["tenant"] == -1)) # number of vacant units is the available supply
        demand = np.sum(neighborhood_chosen == n) # demand for neighborhood n
        cutoff = cutoff_bids[n]

        # minimum sustainable price: WTP of the poorest agent when utility is zero (beta * income)
        # MSP is like the greatest lower bound on prices
        # if price < MSP, then markets should move prices upwards since even the poorest agent can easily afford the current rent
        # if price >= MSP, then this is because agents derive some social utility from staying in the neighborhood
        incomes = agents["income"][agents_mask]
        if incomes.size > 0: # avoid bugs where no residents are in the neighborhood
            P_floor = β * np.min(incomes)
        else:
            P_floor = 0.0

        if demand >= vacancies and cutoff > 0:
            # update the prices of all houses in neighborhood n to be the cutoff price
            new_price = np.full_like(old_price,cutoff)
        else: # cutoff = 0 implies there weren't any bids
            new_price = old_price * decay_rate # slight price decay

        # clip the house prices so that changes aren't too drastic
        clipped = np.clip(new_price, a_min = old_price * (1-max_change), a_max = old_price * (1+max_change))
        houses["value"][mask] = np.maximum(clipped, P_floor) # ensures price >= MSP
    return houses

"--------------------------------- get an array of current rent in each neighborhood --------------------------------"
def get_current_rents(houses, n_neighborhoods=N_NEIGHBORHOODS):
    current_rents = np.zeros(n_neighborhoods)
    for i in range(n_neighborhoods):
        # select all houses in neighborhood i
        mask = houses['neighborhood'] == i
        # since all houses have the same rent, rent of neighborhood i = rent of first house in neighborhood i
        current_rents[i] = houses['value'][np.where(mask)[0][0]]
    return current_rents