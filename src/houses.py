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

    # guard: an agent can only live in one home. if a house lists a tenant who now lives somewhere else, it's vacant
    for h in range(len(houses)):
        t = houses["tenant"][h]
        if t != -1 and agents["house"][t] != h:
            houses["tenant"][h] = -1
    return
"------------------------------- how much each tenant would pay to stay where they are -------------------------------"
# a tenant's stay value is their bid-rent for their current neighborhood (see bid_rent in common.py)
# it's used twice, so evictions and the rent auction follow exactly the same rule:
#   1. check_priced_out: a tenant leaves when the rent passes it (they'd rather be in nonmarket housing)
#   2. clearing_price: it's the tenant's claim on their home in the neighborhood's rent auction
@njit(parallel = True, cache = True)
def stay_values(agents, proportions, q_nm, delta = DELTA):
    n_agents = agents.size
    out = np.zeros(n_agents)
    for i in prange(n_agents):
        if agents["house"][i] == -1:
            continue # agents in nonmarket housing have no home to stay in
        b = agents["income_bracket"][i]
        q = proportions[agents["neighborhood"][i], b]
        out[i] = bid_rent(agents["income"][i], np.float64(agents["theta"][i]), q, q_nm[b], delta)
    return out

"--------------------------------- check if an agent can no longer afford their home --------------------------------"
# home_rents: the rent each home charges (houses["value"], or houses["rent_charged"] under the policy)
def check_priced_out(agents, home_rents, stay):
    housed = agents["house"] >= 0
    rent = np.zeros(agents.size)
    rent[housed] = home_rents[agents["house"][housed]]
    return housed & (rent > stay)

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
# FIX (pricing): the rent signal now comes from a uniform price auction over the WHOLE neighborhood
# every home in the neighborhood is up for grabs, current tenants 'bid' what they'd pay to stay (stay_values) and
# outsiders bid what they'd pay to move in. with H homes, the market clearing rent is the (H+1)-th highest of all
# these bids: the highest bid that doesn't get a home. if there are no more bids than homes, there's no excess demand
# and the signal is 0, so the rent decays
# what this replaced:
#   - originally, a full neighborhood was skipped, so its signal was 0 and its rent FELL no matter how many wanted in
#   - the lowest winning outsider bid ignored the people already living there: a few poor outsiders winning a couple
#     of vacancies could crash the rent of a rich neighborhood, and a single rich outsider bidding on a full one could
#     push everyone's rent up 10% a round. with logit choice spreading bids everywhere, that evicted most of the city
# a tenant is priced out exactly when the rent passes their stay WTP, so this is the same rule check_priced_out applies
@njit(cache = True)
def clearing_price(incumbent_values, outsider_bids, n_homes):
    n_claims = incumbent_values.shape[0] + outsider_bids.shape[0]
    if n_claims <= n_homes:
        return 0.0
    claims = np.concatenate((incumbent_values, outsider_bids))
    claims = np.sort(claims)[::-1]
    return claims[n_homes]

@njit(cache = True)
def outsider_mask(agents, houses, bidders, n):
    # True for bidders who don't currently have a home in neighborhood n
    # (we can't use agents["neighborhood"] for this: agents in nonmarket housing keep their old neighborhood)
    out = np.ones(bidders.shape[0], dtype = np.bool_)
    for j in range(bidders.shape[0]):
        h = agents["house"][bidders[j]]
        if h != -1 and houses["neighborhood"][h] == n:
            out[j] = False
    return out

# what the winners of the vacant homes pay: the uniform price among the outsiders competing for the k vacancies
@njit(cache = True)
def vacancy_price(bids, sorted_bidders, k):
    D = sorted_bidders.shape[0]
    if D == 0 or k == 0:
        return 0.0
    if D > k:
        return bids[sorted_bidders[k]] # highest losing bid
    return bids[sorted_bidders[D-1]] # everyone wins, pays the lowest bid

@njit(cache=True)
def allocate_houses(agents, houses, bids, neighborhood_chosen, stay):
    n_neighborhoods = np.max(houses["neighborhood"])+1
    cutoff_bids = np.zeros(n_neighborhoods) # rent signal per neighborhood, 0 => no excess demand
    vacant_mask = houses["tenant"] == -1
    # FIX: this used to be overwritten in every neighborhood, so it only ever held the last neighborhood's winners
    num_winners = 0

    # run an auction in each neighborhood
    for n in range(n_neighborhoods):
        in_n = houses["neighborhood"] == n
        homes = np.where(in_n)[0]
        bidders = np.where(neighborhood_chosen == n)[0]

        # rent signal from the whole neighborhood (see clearing_price)
        tenants = houses["tenant"][homes]
        tenants = tenants[tenants != -1]
        outsiders = bidders[outsider_mask(agents, houses, bidders, n)] # bidders who don't already live here
        cutoff_bids[n] = clearing_price(stay[tenants], bids[outsiders], homes.shape[0])

        if bidders.shape[0] == 0:
            continue
        vacancies = np.where(vacant_mask & in_n)[0]
        k = vacancies.shape[0]
        if k == 0:
            continue # nobody moves in, but the rent signal can still push the rent up

        # sort bids in descending order
        order = np.argsort(-bids[bidders]) # -> argsort gives indices
        sorted_bidders = bidders[order] # use those indices to sort here

        # FIX: one rent per neighborhood. winners used to pay their own auction price while everyone else in the
        # neighborhood faced the (often higher) prevailing rent, so a winner could be priced out the very next round.
        # now a winner must bid at least the prevailing rent, and pays it; update_prices moves the rent afterwards
        # the exception is a completely empty neighborhood (round 0): it has no rent yet, so the auction discovers it
        if k == homes.shape[0]:
            price = vacancy_price(bids, sorted_bidders, k)
            for h in homes:
                houses["value"][h] = price
        else:
            price = houses["value"][homes[0]]
        can_pay = sorted_bidders[bids[sorted_bidders] >= price]
        winners = can_pay[:k] # -> anyone beyond k (num vacancies) automatically loses
        num_winners += winners.shape[0]

        # give winners their vacant homes, all at the same price
        for w, v in zip(winners, vacancies):
            # FIX: a housed agent who wins a home elsewhere used to stay listed as the tenant of their old home too,
            # so every move silently destroyed a home (a 'ghost' tenancy). vacate the old home first
            old_home = agents["house"][w]
            if old_home != -1:
                houses["tenant"][old_home] = -1
            houses["tenant"][v] = w
            agents["house"][w] = v
            agents["neighborhood"][w] = n
            agents["nonmarket_housing"][w] = False
            agents["rent_paid"][w] = price

    agent_house_mapping(agents, houses) # update the mapping after the allocation is made
    return agents, houses, cutoff_bids, num_winners

"-------------------------------------- update the house prices based on demand -------------------------------------"
# every home in a neighborhood shares one rent, and it moves by at most max_change in one round:
#   excess demand (more claims than homes in the clearing auction): move towards the clearing rent
#   excess supply (vacant market homes, no excess demand): decay
#   neither (full, and nobody outbid the tenants): hold the rent where it is
# FIX: the rent used to decay whenever there was no excess demand, including in full neighborhoods. since agents only
# bid where there's a vacancy, a full neighborhood never sees outside bids, so its rent fell 5% every round forever and
# rents ended up at 0.2-2% of income
# the old price floor (beta * poorest resident's income) is gone along with beta
def update_prices(houses, cutoff_bids,
                  decay_rate = DECAY_RATE, # fall in price if supply > demand
                  max_change = MAX_CHANGE): # maximum % change in price in one round
    n_neighborhoods = np.max(houses["neighborhood"]) + 1
    vacant = houses["tenant"] == -1
    if "low_rent" in houses.dtype.names:
        vacant = vacant & ~houses["low_rent"] # empty set aside homes aren't market supply
    n_vacant = np.bincount(houses["neighborhood"][vacant], minlength = n_neighborhoods)
    for n in range(n_neighborhoods):
        mask = houses["neighborhood"] == n
        old_price = houses["value"][mask][0]
        cutoff = cutoff_bids[n]
        if cutoff > 0: # excess demand: move towards the auction's market clearing rent
            new_price = cutoff
        elif n_vacant[n] > 0: # excess supply
            new_price = old_price * decay_rate
        else: # full, no excess demand: equilibrium
            continue
        # clip the change so rents don't swing wildly in one round
        houses["value"][mask] = min(max(new_price, old_price * (1-max_change)), old_price * (1+max_change))
    return houses