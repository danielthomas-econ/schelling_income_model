from .common import *
import numpy as np
from numba import njit

"------------------------------------------ which neighborhoods can you move into? -----------------------------------"
def vacant_neighborhoods(houses, n_neighborhoods = N_NEIGHBORHOODS, low_rent_only = False, market_only = False):
    # boolean array: does neighborhood k have at least one vacant home (of the requested kind)?
    vacant = houses["tenant"] == -1
    if low_rent_only:
        vacant = vacant & houses["low_rent"]
    if market_only:
        vacant = vacant & ~houses["low_rent"]
    counts = np.bincount(houses["neighborhood"][vacant], minlength = n_neighborhoods)[:n_neighborhoods]
    return counts > 0

"--------------------------------------------- rents agents see when bidding -----------------------------------------"
def bidding_rents(houses, n_neighborhoods = N_NEIGHBORHOODS, low_rent = False):
    # the rent you'd pay in each neighborhood: its single market rent (or the discounted rent for set aside homes)
    # a completely empty neighborhood (round 0) has no rent yet: 0, and the auction discovers it
    field = "rent_charged" if low_rent else "value"
    rents = np.zeros(n_neighborhoods)
    occupied = np.bincount(houses["neighborhood"][houses["tenant"] != -1], minlength = n_neighborhoods)[:n_neighborhoods]
    for n in range(n_neighborhoods):
        if occupied[n] == 0:
            continue
        if low_rent:
            idx = np.where((houses["neighborhood"] == n) & houses["low_rent"])[0]
        else:
            idx = np.where(houses["neighborhood"] == n)[0]
            if "low_rent" in houses.dtype.names:
                idx = idx[~houses["low_rent"][idx]]
        if idx.size > 0:
            rents[n] = houses[field][idx[0]]
    return rents

"--------------------------------------------------- bidding logic --------------------------------------------------"
# every unhappy agent picks ONE neighborhood and bids their bid-rent r* for it (see common.py)
# a neighborhood k is an option only if:
#   1. it has a vacant home the agent could rent
#   2. its current rent is below r*_k, i.e. the agent prefers it to nonmarket housing at that rent
#   3. for agents who already have a home: U_k > U at their current home (they only move somewhere better)
# among the options, the choice is logit over U (see CHOICE_TEMPERATURE). the bid is r*_k, which is always >= the rent
@njit(cache = True)
def _place_bid_kernel(happy, incomes, thetas, brackets, eligible, current_nb, current_rent,
                      proportions, q_nm, rents_market, rents_eligible, available_market, available_eligible,
                      delta, temperature):
    n_agents = incomes.size
    n_neighborhoods = proportions.shape[0]
    bids = np.zeros(n_agents, dtype = np.float64)
    # which neighborhood the agents chooses to bid for, -1 => they're not bidding this round
    neighborhood_chosen = np.full(n_agents, -1, dtype = np.int64)
    u = np.zeros(n_neighborhoods) # utility of each option (0 => not an option)
    r_star = np.zeros(n_neighborhoods) # bid-rent for each neighborhood

    for i in range(n_agents):
        if happy[i]: # happy agents stay put
            continue
        y = incomes[i]
        theta = np.float64(thetas[i])
        b = brackets[i]
        if eligible[i]:
            rents = rents_eligible
            available = available_eligible
        else:
            rents = rents_market
            available = available_market

        # utility where they live now (0 for agents in nonmarket housing: condition 2 already covers them)
        u_current = 0.0
        if current_nb[i] >= 0:
            u_current = utility(y, theta, proportions[current_nb[i], b], current_rent[i], delta)

        u_best = 0.0
        for k in range(n_neighborhoods):
            u[k] = 0.0
            if not available[k]:
                continue
            r_star[k] = bid_rent(y, theta, proportions[k, b], q_nm[b], delta)
            if rents[k] >= r_star[k]: # not better than nonmarket housing at this rent
                continue
            uk = utility(y, theta, proportions[k, b], rents[k], delta)
            if uk <= u_current: # not better than where they live now
                continue
            u[k] = uk
            if uk > u_best:
                u_best = uk
        if u_best <= 0.0: # nowhere better to go this round
            continue

        # logit choice over the options, utilities normalized so the best option = 1
        total = 0.0
        for k in range(n_neighborhoods):
            if u[k] > 0.0:
                u[k] = np.exp((u[k] / u_best - 1.0) / temperature)
                total += u[k]
        draw = np.random.random() * total
        chosen = -1
        running = 0.0
        for k in range(n_neighborhoods):
            if u[k] > 0.0:
                chosen = k
                running += u[k]
                if running >= draw:
                    break

        bids[i] = r_star[chosen]
        neighborhood_chosen[i] = chosen

    return bids, neighborhood_chosen

def place_bid(agents, proportions, q_nm, home_rents,
              rents_market, available_market,
              rents_eligible = None, # rents eligible agents face (policy only)
              available_eligible = None, # neighborhoods eligible agents can move into (policy only)
              delta = DELTA,
              temperature = CHOICE_TEMPERATURE):
    # q_nm: nonmarket housing quality per bracket (nonmarket_quality_by_bracket)
    # home_rents: the rent paid for each home (houses["value"], or houses["rent_charged"] under the policy)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if rents_eligible is None:
        rents_eligible = rents_market
    if available_eligible is None:
        available_eligible = available_market
    if "low_rent" in agents.dtype.names:
        eligible = agents["low_rent"]
    else:
        eligible = np.zeros(agents.size, dtype = np.bool_)

    # where each agent lives now and what they pay (-1 / 0 if in nonmarket housing; their 'neighborhood' field is stale then)
    housed = agents["house"] >= 0
    current_nb = np.where(housed, agents["neighborhood"].astype(np.int64), -1)
    current_rent = np.zeros(agents.size)
    current_rent[housed] = home_rents[agents["house"][housed]]

    return _place_bid_kernel(agents["happy"], agents["income"], agents["theta"], agents["income_bracket"], eligible,
                             current_nb, current_rent, proportions, q_nm,
                             np.asarray(rents_market, dtype = np.float64), np.asarray(rents_eligible, dtype = np.float64),
                             available_market, available_eligible, float(delta), float(temperature))
