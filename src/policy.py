from numba import jit, njit, prange
import numpy as np
import time
from .houses import *
from .common import *
from .agents import *
from .stats import *
from .bidding import *
from .sim import *
from .plots import plot_run_summary, plot_mc_summary, plot_policy_comparison, plot_policy_bracket_happiness, plot_welfare
from scipy.stats import t as student_t

# i've copy pasted a lot of the code from the regular sim here to make the changes for low rent
# to not force the original sim code to have a lot of 'if affordable housing policy' lines

"--------------------------------- generate agents with a new low rent housing field --------------------------------"
def generate_agents_affordable(n_agents = N_AGENTS):
    # gen a structured array to store everything
    # low level memory optimization
    # just for 9 columns, it has bought down memory consumption by ~82% vs the list of arrays structure
    agent_dtype = np.dtype([
        ("id", np.int32),
        ("income", np.float64),
        ("income_bracket", np.uint8),
        ("neighborhood", np.int8), # int8 works with the -1 neighborhood assignment
        ("happy", np.bool_),
        ("house", np.int32),
        ("nonmarket_housing", np.bool_), # does the agent have a real house or do they live in nonmarket housing (proxy for homelessness kinda)
        ("rent_paid", np.float64), # no need for checking tenancy, rent_paid = 0 => not a tenant
        ("theta", np.float32), # numba doesnt like float16, so we stick to float32 here
        ("low_rent", np.bool_), # is this guy eligible for low rent?
    ])

    # initialize agents
    agents = np.zeros(n_agents, dtype=agent_dtype)

    agents["id"] = np.arange(n_agents)
    agents["income"] = get_incomes(agents)
    agents["income_bracket"] = find_income_brackets(agents)
    agents["neighborhood"] = allocate_neighborhood(agents)
    agents["happy"] = False # initially everyone is depressed :(
    agents["house"] = np.full(n_agents,-1) # ASSIGN HOUSES INITIALLY TOO
    agents["nonmarket_housing"] = True
    agents["rent_paid"] = np.zeros(n_agents)
    agents["theta"] = np.random.uniform(THETA_MIN, THETA_MAX, n_agents)
    return agents

"------------------------------------- generate houses with a field for low rent ------------------------------------"
def initialize_houses_affordable(agents, starting_house_price = STARTING_HOUSE_PRICE):
    n_agents = agents.size

    # note: we use rent charged to let the actual market price be decided by the 'value' field, but charge agents only
    # 'rent_charged' so that the policy actualy works
    houses_datatype = np.dtype([
        ("id", np.int32),
        ("tenant", np.int32), # which agent lives here?
        ("neighborhood", np.uint8),
        ("value", np.float64), # how much the house is actually worth, not the same as rent_paid by agent
        ("rent_charged", np.float64), # how much do the people actually get charged?
        ("low_rent", np.bool_), # is this house considered affordable?
    ])

    houses = np.zeros(n_agents, houses_datatype)
    houses["id"] = np.arange(n_agents)
    houses["tenant"] = np.full(n_agents, -1)
    houses["neighborhood"] = allocate_neighborhood(agents)
    houses["value"] = np.full(n_agents, starting_house_price)
    houses["rent_charged"] = np.full(n_agents, starting_house_price)
    return houses

"----------------------------- decide which income brackets qualify for low rent housing ----------------------------"
@jit(parallel = True, cache = True)
def check_low_rent_eligibility(agents, income_cutoff = 2): # cutoff of 2 makes 30% eligible
    for i in prange(len(agents)):
        agents["low_rent"][i] = (agents["income_bracket"][i] <= income_cutoff)
    return agents

"--------------------------------- decide which houses qualify for low rent housing ---------------------------------"
# plain numpy now: the fancy-index assignment inside a parallel numba loop was fragile and this only runs once per city
def assign_low_rent_house(houses, houses_eligible = 0.2): # what share of each neighborhood's homes are set aside
    houses["low_rent"] = False
    for i in range(np.max(houses["neighborhood"])+1):
        idx = np.where(houses["neighborhood"] == i)[0] # lets us work with indices instead of typical boolean array
        k = int(houses_eligible * len(idx)) # how many houses are eligible?
        houses["low_rent"][idx[:k]] = True # we set the first k houses in our neighborhood to be low rent ones
    return houses

"--------------------------------------- set rent charged from the market value -------------------------------------"
def set_rent_charged(houses, lower_price = 0.6):
    # market homes charge the market value, set aside homes charge (market value) * lower_price
    houses["rent_charged"] = houses["value"]
    houses["rent_charged"][houses["low_rent"]] *= lower_price
    return houses

"------------------------------------------ draw a fresh city with the policy ----------------------------------------"
# uses exactly the same sequence of random draws as new_population in sim.py, so with the same seed the
# baseline and the policy run start from the same city (common random numbers)
def new_population_affordable(n_agents = N_AGENTS,
                              starting_house_price = STARTING_HOUSE_PRICE,
                              theta_min = THETA_MIN,
                              theta_max = THETA_MAX,
                              income_cutoff = 2,
                              houses_eligible = 0.2,
                              lower_price = 0.6):
    agents = generate_agents_affordable(n_agents)
    agents["theta"] = np.random.uniform(theta_min, theta_max, n_agents)
    houses = initialize_houses_affordable(agents)
    houses["value"] = np.full(n_agents, starting_house_price)

    # assign eligible agents and houses
    agents = check_low_rent_eligibility(agents, income_cutoff)
    houses = assign_low_rent_house(houses, houses_eligible)
    # FIX: rent_charged used to start at the full price for every home, so the discount only kicked in after round 0
    houses = set_rent_charged(houses, lower_price)
    return agents, houses

"------------------------------------ allocate houses with a targeted set aside -------------------------------------"
# FIX: the policy used to reuse allocate_houses, which ignores both low_rent flags. so anyone who won a set aside home
# got the discount, and since homes go to the highest bids, the discount mostly went to richer agents
# now each neighborhood runs two auctions:
#   1. set aside homes: only eligible bidders, highest bids first, and the bid must cover the discounted rent
#   2. market homes: everyone left over (ineligible bidders + eligible ones who didn't get a set aside home),
#      same uniform price auction as allocate_houses. only this auction sets the market clearing price,
#      so the discounted units don't drag down the market rent
@njit(cache = True)
def allocate_houses_affordable(agents, houses, bids, neighborhood_chosen, stay):
    n_neighborhoods = np.max(houses["neighborhood"])+1
    cutoff_bids = np.zeros(n_neighborhoods) # market rent signal, see clearing_price in houses.py
    vacant_mask = houses["tenant"] == -1
    num_winners = 0 # total across all neighborhoods, both auctions

    for n in range(n_neighborhoods):
        in_n = houses["neighborhood"] == n
        homes = np.where(in_n)[0]
        bidders = np.where(neighborhood_chosen == n)[0]
        order = np.argsort(-bids[bidders])
        sorted_bidders = bidders[order] # sorted by bid, descending
        placed = np.zeros(sorted_bidders.shape[0], dtype=np.bool_)

        # a completely empty neighborhood (round 0) has no rent yet: the auction over all its homes discovers it,
        # and the set aside homes get the same discount on it
        if homes.shape[0] > 0 and bidders.shape[0] > 0 and np.all(vacant_mask[homes]):
            p0 = vacancy_price(bids, sorted_bidders, homes.shape[0])
            for h in homes:
                ratio = houses["rent_charged"][h] / houses["value"][h] if houses["value"][h] > 0 else 1.0
                houses["value"][h] = p0
                houses["rent_charged"][h] = p0 * ratio

        # ---------- auction 1: set aside homes, eligible agents only ----------
        lr_vacancies = np.where(vacant_mask & in_n & houses["low_rent"])[0]
        j = 0 # how many set aside homes we've filled
        for b in range(sorted_bidders.shape[0]):
            if j >= lr_vacancies.shape[0]:
                break
            w = sorted_bidders[b]
            if not agents["low_rent"][w]:
                continue
            v = lr_vacancies[j]
            # all homes in a neighborhood share one price, so once one eligible bid can't cover the discounted rent,
            # no lower bid can either
            if bids[w] < houses["rent_charged"][v]:
                break
            old_home = agents["house"][w] # FIX: vacate the winner's old home (no ghost tenancies)
            if old_home != -1:
                houses["tenant"][old_home] = -1
            houses["tenant"][v] = w
            agents["house"][w] = v
            agents["neighborhood"][w] = n
            agents["nonmarket_housing"][w] = False
            agents["rent_paid"][w] = houses["rent_charged"][v]
            placed[b] = True
            j += 1
        num_winners += j

        # ---------- market rent signal: market homes only ----------
        # the clearing auction runs over the neighborhood's market homes: their tenants' stay values vs the bids of
        # everyone who's left and doesn't already live here. set aside homes have their own (discounted) rent
        market_homes = np.where(in_n & np.logical_not(houses["low_rent"]))[0]
        tenants = houses["tenant"][market_homes]
        tenants = tenants[tenants != -1]
        remaining = sorted_bidders[np.logical_not(placed)] # still sorted by bid
        outsiders = remaining[outsider_mask(agents, houses, remaining, n)]
        cutoff_bids[n] = clearing_price(stay[tenants], bids[outsiders], market_homes.shape[0])

        # ---------- auction 2: market homes, everyone who's left ----------
        mk_vacancies = np.where(vacant_mask & in_n & np.logical_not(houses["low_rent"]))[0]
        k = mk_vacancies.shape[0]
        if k == 0 or remaining.shape[0] == 0:
            continue
        # one rent per neighborhood: winners must bid at least the prevailing market rent, and pay it
        price = houses["value"][mk_vacancies[0]]
        can_pay = remaining[bids[remaining] >= price]
        winners = can_pay[:k]
        num_winners += winners.shape[0]

        for w, v in zip(winners, mk_vacancies):
            old_home = agents["house"][w] # FIX: vacate the winner's old home (no ghost tenancies)
            if old_home != -1:
                houses["tenant"][old_home] = -1
            houses["tenant"][v] = w
            agents["house"][w] = v
            agents["neighborhood"][w] = n
            agents["nonmarket_housing"][w] = False
            agents["rent_paid"][w] = price

    agent_house_mapping(agents, houses)
    return agents, houses, cutoff_bids, num_winners

"-------------------------------------- pricing system under affordable housing -------------------------------------"
# same rule as update_prices in houses.py for the market rent, then the discount for set aside homes
def update_prices_affordable(houses, cutoff_bids,
                             decay_rate = DECAY_RATE, # fall in price if supply > demand
                             max_change = MAX_CHANGE, # maximum % change in price in one round
                             lower_price = 0.6): # low rent homes cost (market rent) * (lower_price), 60% by default
    houses = update_prices(houses, cutoff_bids, decay_rate, max_change)
    houses = set_rent_charged(houses, lower_price)
    return houses

"-------------------- bookkeeping for the rent paid since its not the same as cutoff bids anymore -------------------"
@njit(parallel=True, cache=True)
def update_rent_paid_affordable(agents, houses):
    for i in prange(agents.size):
        h = agents["house"][i]
        if h == -1:
            agents["rent_paid"][i] = 0.0
        else:
            agents["rent_paid"][i] = houses["rent_charged"][h]
    return agents

"----------------------------------- one round of the sim under the policy -----------------------------------------"
# same steps as run_round in sim.py. the differences: rents are what tenants are actually charged (rent_charged),
# eligible agents see the discounted rent where a set aside home is free, and the allocation runs two auctions
def run_round_affordable(agents, houses, happiness_percent = DEFAULT_HAPPINESS_PERCENT, delta = DELTA,
                         nonmarket_quality = NONMARKET_QUALITY, temperature = CHOICE_TEMPERATURE, lower_price = 0.6,
                         preference = PREFERENCE, homophily_window = HOMOPHILY_WINDOW):
    proportions, q_nm = neighborhood_quality(agents, nonmarket_quality, preference, homophily_window)
    agents = check_happiness(agents, proportions, happiness_percent)

    # set aside tenants are judged on the discounted rent they actually pay
    stay = stay_values(agents, proportions, q_nm, delta)
    evict_priced_out(agents, houses, check_priced_out(agents, houses["rent_charged"], stay))

    # rents agents face when choosing where to bid (0 for an empty neighborhood: the auction discovers its rent)
    # eligible agents see the discounted rent wherever a set aside home is free, the market rent elsewhere
    rents_market = bidding_rents(houses)
    rents_eligible = np.where(vacant_neighborhoods(houses, low_rent_only = True),
                              bidding_rents(houses, low_rent = True), rents_market)
    # ineligible agents can only move where a market home is free, eligible ones anywhere with a free home of either kind
    bids, neighborhoods_chosen = place_bid(agents, proportions, q_nm, houses["rent_charged"],
                                           rents_market = rents_market,
                                           available_market = vacant_neighborhoods(houses, market_only = True),
                                           rents_eligible = rents_eligible,
                                           available_eligible = vacant_neighborhoods(houses),
                                           delta = delta, temperature = temperature)
    agents, houses, cutoff_bids, num_winners = allocate_houses_affordable(agents, houses, bids, neighborhoods_chosen, stay)
    houses = update_prices_affordable(houses, cutoff_bids, lower_price = lower_price)

    # use the correct way to update rent paid
    agents = update_rent_paid_affordable(agents, houses)
    return agents, houses, bids, num_winners

"------------------------------- who actually lives in the set aside homes? (sanity check) ---------------------------"
def set_aside_targeting(agents, houses):
    occupied = houses["low_rent"] & (houses["tenant"] != -1)
    tenants = houses["tenant"][occupied]
    n_occupied = tenants.size
    n_eligible = int(np.sum(agents["low_rent"][tenants])) if n_occupied > 0 else 0
    return {
        "set_aside_homes": int(np.sum(houses["low_rent"])),
        "occupied": n_occupied,
        "occupied_by_eligible": n_eligible, # should always equal 'occupied' now
    }

"--------------------------------- the affordable housing policy version of the sim ---------------------------------"
def sim_one_round_affordable(n_agents = N_AGENTS,
                  n_neighborhoods = N_NEIGHBORHOODS,
                  max_rounds = 100,
                  happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                  starting_house_price = STARTING_HOUSE_PRICE,
                  nonmarket_quality = NONMARKET_QUALITY,
                  temperature = CHOICE_TEMPERATURE,
                  delta = DELTA,
                  theta_min = THETA_MIN,
                  theta_max = THETA_MAX,
                  preference = PREFERENCE, # "status" or "homophily", see common.py
                  homophily_window = HOMOPHILY_WINDOW,
                  converge = False,
                  convergence_bound = 5, # will the sim end if we have churn convergence?
                  income_cutoff = 2, # agents with this income bracket and below are eligible
                  houses_eligible = 0.2, # what percent of houses in each neighborhood are 'affordable'
                  lower_price = 0.6, # low rent homes will cost (actual rent) * (lower_price), 60% by default
                  seed = None,
                  plot = True,
                  ):
    start = time.time()
    set_seed(seed)
    agents, houses = new_population_affordable(n_agents, starting_house_price, theta_min, theta_max,
                                               income_cutoff, houses_eligible, lower_price)

    stats = initialize_stats(num_rounds=max_rounds, num_neighborhoods=n_neighborhoods)
    count = 0 # tracks iterations
    prev_house = None # we initialize previous houses = none since ofc its not defined yet

    # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
    while not np.all(agents["happy"]):
        print(f"Round {count}")
        agents, houses, bids, num_winners = run_round_affordable(agents, houses, happiness_percent, delta,
                                                                 nonmarket_quality, temperature, lower_price,
                                                                 preference, homophily_window)
        print(f"Happiness: {(np.sum(agents["happy"])*100)/n_agents:.3f}%")
        print()

        # log the data
        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        stats["num_bids"][count] = np.count_nonzero(bids)
        stats["winning_bids"][count] = num_winners

        count += 1
        if should_stop(stats["churn"], count, max_rounds, converge, convergence_bound):
            break
    last_round = count-1

    print(f"Set aside targeting: {set_aside_targeting(agents, houses)}")

    if plot:
        plot_run_summary(stats, agents, last_round, title_suffix = " (affordable housing policy)")

    end = time.time()
    print(f"Time taken: {end-start:.4f} seconds")
    return agents, houses, stats, last_round

"--------------------------------- monte carlo version of the affordable policy sim ---------------------------------"
def monte_carlo_sim_affordable(n_agents = N_AGENTS,
                    n_neighborhoods = N_NEIGHBORHOODS,
                    max_rounds = 100,
                    n_runs = 30,
                    happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                    starting_house_price = STARTING_HOUSE_PRICE,
                    nonmarket_quality = NONMARKET_QUALITY,
                    temperature = CHOICE_TEMPERATURE,
                    delta = DELTA,
                    theta_min = THETA_MIN,
                    theta_max = THETA_MAX,
                    preference = PREFERENCE, # "status" or "homophily", see common.py
                    homophily_window = HOMOPHILY_WINDOW,
                    converge = False,
                    convergence_bound = 5, # will the sim end if we have churn convergence?
                    income_cutoff = 2, # agents with this income bracket and below are eligible
                    houses_eligible = 0.2, # what percent of houses in each neighborhood are 'affordable'
                    lower_price = 0.6, # low rent homes will cost (actual rent) * (lower_price), 60% by default
                    seed = None, # run r uses seed + r; pass the same seed as monte_carlo_sim to compare like with like
                    redraw_population = True,
                    plot = True,
                    on_run_end = None, # optional function(run_id, agents, houses), called with each run's final state
                    ):
    start = time.time()
    set_seed(seed)
    agents_og, houses_og = new_population_affordable(n_agents, starting_house_price, theta_min, theta_max,
                                                     income_cutoff, houses_eligible, lower_price)

    mc_stats = initialize_mc_stats(num_runs = n_runs, num_rounds=max_rounds, num_agents=agents_og.size, num_neighborhoods=n_neighborhoods)

    final_happiness_by_bracket = mc_stats["bracket_happiness"]

    for current_run in range(n_runs):
        print(f"Running run {current_run+1}")
        print()
        if seed is not None:
            set_seed(seed + current_run)
        if redraw_population and current_run > 0:
            agents, houses = new_population_affordable(n_agents, starting_house_price, theta_min, theta_max,
                                                       income_cutoff, houses_eligible, lower_price)
        else:
            agents = agents_og.copy()
            houses = houses_og.copy()
        count = 0 # tracks iterations

        # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
        while not np.all(agents["happy"]):
            agents, houses, bids, num_winners = run_round_affordable(agents, houses, happiness_percent, delta,
                                                                     nonmarket_quality, temperature, lower_price,
                                                                     preference, homophily_window)

            # log the data
            mc_stats = get_mc_stats(mc_stats, agents, houses, run_id = current_run, current_round=count)
            mc_stats["num_bids"][current_run, count] = np.count_nonzero(bids)
            mc_stats["winning_bids"][current_run, count] = num_winners

            count += 1
            if should_stop(mc_stats["churn"][current_run], count, max_rounds, converge, convergence_bound):
                break

        mc_stats["last_round"][current_run] = count-1

        # happiness by income bracket at the end of the run
        for i in range(N_BRACKETS):
            mask = agents["income_bracket"] == i
            total_i = np.sum(mask)
            if total_i > 0:
                final_happiness_by_bracket[current_run, i] = np.sum(agents["happy"][mask]) * 100 / total_i

        # lets evaluate_policy read each run's final state (used for the welfare comparison)
        if on_run_end is not None:
            on_run_end(current_run, agents, houses)

    last_round = int(np.max(mc_stats["last_round"]))
    print(f"Set aside targeting (last run): {set_aside_targeting(agents, houses)}")

    if plot:
        plot_mc_summary(mc_stats, last_round, title_suffix = " (affordable housing policy)")

    end = time.time()
    print(f"Total time taken: {end-start:.4f} secs")
    return agents, houses, mc_stats, last_round

"------------------------------- paired policy effect at each run's final round ------------------------------------"
def paired_difference(mc_b, mc_p, stat, confidence = 95):
    runs = np.arange(mc_b["last_round"].size)
    b = mc_b[stat][runs, mc_b["last_round"]].astype(np.float64)
    p = mc_p[stat][runs, mc_p["last_round"]].astype(np.float64)
    d = p - b
    d = d[np.isfinite(d)]
    if d.size < 2:
        return None
    alpha = (100 - confidence) / 100
    half = student_t.ppf(1 - alpha/2, d.size - 1) * np.std(d, ddof=1) / np.sqrt(d.size)
    return np.mean(d), np.mean(d) - half, np.mean(d) + half

"--------------------------------------------- welfare: each agent's end state ----------------------------------------"
# each agent's realized situation at the end of a run: neighborhood quality q, budget share left after rent c, utility u
# an agent whose home is now worse than nonmarket housing would leave next round, so their welfare is nonmarket housing's
# (q = q_nm, c = 1). note c is a SHARE of the housing budget, so utility doesn't grow with income by itself
def welfare_state(agents, houses, home_rents, delta = DELTA, nonmarket_quality = NONMARKET_QUALITY,
                  preference = PREFERENCE, homophily_window = HOMOPHILY_WINDOW):
    proportions, q_nm = neighborhood_quality(agents, nonmarket_quality, preference, homophily_window)

    b = agents["income_bracket"]
    y = agents["income"]
    theta = agents["theta"].astype(np.float64)
    housed = agents["house"] >= 0

    q = q_nm[b].copy() # start everyone in nonmarket housing, then fill in the housed
    c = np.ones(agents.size)
    q[housed] = proportions[agents["neighborhood"][housed], b[housed]]
    rent = home_rents[agents["house"][housed]]
    c[housed] = np.clip((delta * y[housed] - rent) / (delta * y[housed]), 0.0, None)

    u = q ** theta * c ** (1.0 - theta)
    u_nm = q_nm[b] ** theta
    worse = u < u_nm # better off in nonmarket housing: that's where they'd go
    q[worse] = q_nm[b][worse]
    c[worse] = 1.0
    u[worse] = u_nm[worse]
    return q, c, u

"------------------------------------ welfare: equivalent variation, % of income -------------------------------------"
# how much is the policy worth to an agent, measured in their BASELINE situation?
# find the budget share c_eq that, at their baseline neighborhood quality, gives them their policy utility:
#   q_base^theta * c_eq^(1-theta) = u_policy  =>  c_eq = (u_policy / q_base^theta)^(1 / (1-theta))
# EV = delta * (c_eq - c_base) is that change as a share of income: +0.02 means the policy is worth as much to them as
# a rent cut of 2% of their income, -0.02 as a rent rise of 2%. because it's a share of income, rich and poor agents
# are measured in the same unit and each counts equally when averaged (a rupee measure would be dominated by the rich)
def equivalent_variation(q_base, c_base, u_policy, theta, delta = DELTA):
    with np.errstate(divide = "ignore", invalid = "ignore"):
        c_eq = (u_policy / q_base ** theta) ** (1.0 / (1.0 - theta))
    return delta * (c_eq - c_base)

# rent as a share of income for housed agents, and the share of each bracket that's housed
def housing_by_bracket(agents, home_rents):
    housed = agents["house"] >= 0
    rent_share = np.full(agents.size, np.nan)
    rent_share[housed] = home_rents[agents["house"][housed]] / agents["income"][housed]
    housed_pct = np.full(N_BRACKETS, np.nan)
    mean_rent_share = np.full(N_BRACKETS, np.nan)
    for i in range(N_BRACKETS):
        m = agents["income_bracket"] == i
        if np.any(m):
            housed_pct[i] = np.mean(housed[m]) * 100
        if np.any(m & housed):
            mean_rent_share[i] = np.mean(rent_share[m & housed]) * 100
    return housed_pct, mean_rent_share

# where each agent ends up under the policy. this shows whether the people who actually get a set aside home gain
WELFARE_GROUPS = ["Eligible, in a set-aside home",
                  "Eligible, in a market home",
                  "Eligible, in nonmarket housing",
                  "Not eligible, housed",
                  "Not eligible, in nonmarket housing"]
def welfare_group_masks(agents, houses):
    housed = agents["house"] >= 0
    in_set_aside = np.zeros(agents.size, dtype = np.bool_)
    in_set_aside[housed] = houses["low_rent"][agents["house"][housed]]
    eligible = agents["low_rent"]
    return [eligible & in_set_aside,
            eligible & housed & ~in_set_aside,
            eligible & ~housed,
            ~eligible & housed,
            ~eligible & ~housed]

def _show_table(df):
    # renders as a formatted table in Jupyter, plain text anywhere else
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())

"------------------------------------ compare the policy outcomes to the baseline -----------------------------------"
# the metrics we're gonna look at to evaluate our policy
POLICY_METRICS = [
    ("happiness", "Happiness (%)"),
    ("nonmarket_housing", "Nonmarket housing (%)"),
    ("churn", "Churn (%)"),
    ("gini", "Gini"),
    ("theil_between", "Theil (between)")
]

def evaluate_policy(n_agents=N_AGENTS,
                    n_neighborhoods=N_NEIGHBORHOODS,
                    max_rounds=100,
                    n_runs=30,
                    confidence=95,
                    band="mean", # "mean" => CI of the mean across runs, "spread" => middle 95% of individual runs
                    seed=0, # same seed for both => run r of the baseline and run r of the policy start from the same city
                    income_cutoff = 2,
                    houses_eligible = 0.2,
                    lower_price = 0.6,
                    plot = True,
                    **kwargs
                    ):

    # ---------- welfare bookkeeping ----------
    # run r of the baseline and run r of the policy start from the same city, so agent i is the same person in both.
    # the baseline stores each agent's end state (q, c), and the policy run compares against it agent by agent
    # memory: 2 float32 per agent per run, e.g. 1m agents x 30 runs = 240 MB
    if seed is None:
        raise ValueError("evaluate_policy needs a seed: the welfare comparison pairs each agent across the two runs")
    delta = kwargs.get("delta", DELTA)
    nonmarket_quality = kwargs.get("nonmarket_quality", NONMARKET_QUALITY)
    preference = kwargs.get("preference", PREFERENCE)
    homophily_window = kwargs.get("homophily_window", HOMOPHILY_WINDOW)
    base_q = np.zeros((n_runs, n_agents), dtype = np.float32)
    base_c = np.zeros((n_runs, n_agents), dtype = np.float32)
    base_income_sum = np.zeros(n_runs)
    per_bracket = ["ev_mean", "ev_median", "gain_pct", "lose_pct",
                   "housed_base", "housed_policy", "rent_share_base", "rent_share_policy"]
    welfare = {k: np.full((n_runs, N_BRACKETS), np.nan) for k in per_bracket}
    welfare["ev_all"] = np.full(n_runs, np.nan) # equal-weighted mean EV over all agents (capped, see compare_policy)
    welfare["ev_median_all"] = np.full(n_runs, np.nan) # median EV over all agents
    welfare["gain_all"] = np.full(n_runs, np.nan) # % of all agents better off
    welfare["lose_all"] = np.full(n_runs, np.nan) # % of all agents worse off
    # same breakdown by where agents end up under the policy (see WELFARE_GROUPS)
    for k in ["group_share", "group_gain", "group_lose", "group_ev_median"]:
        welfare[k] = np.full((n_runs, len(WELFARE_GROUPS)), np.nan)

    def store_baseline(r, agents, houses):
        q, c, _ = welfare_state(agents, houses, houses["value"], delta, nonmarket_quality,
                                preference, homophily_window)
        base_q[r] = q
        base_c[r] = c
        base_income_sum[r] = np.sum(agents["income"])
        welfare["housed_base"][r], welfare["rent_share_base"][r] = housing_by_bracket(agents, houses["value"])

    def compare_policy(r, agents, houses):
        if not np.isclose(np.sum(agents["income"]), base_income_sum[r]):
            raise RuntimeError(f"run {r}: the baseline and policy runs started from different cities")
        _, _, u_p = welfare_state(agents, houses, houses["rent_charged"], delta, nonmarket_quality,
                                 preference, homophily_window)
        theta = agents["theta"].astype(np.float64)
        ev = equivalent_variation(base_q[r].astype(np.float64), base_c[r].astype(np.float64), u_p, theta, delta) * 100
        tol = 1e-4 # in % of income, ignores floating point noise for agents whose situation didn't change
        # means are taken over EV capped at +-(delta * 100)% of income, i.e. no agent's gain or loss counts as more than
        # their whole housing budget. why: an agent with a high theta values neighborhood quality so much that matching a
        # big quality gain with money alone takes many times their income (the utility ratio enters to the power
        # 1/(1-theta), = 10 at theta = 0.9), and a handful of such agents would otherwise dominate the mean.
        # medians and the shares of winners and losers don't need the cap
        ev_capped = np.clip(ev, -delta * 100, delta * 100)
        welfare["ev_all"][r] = np.nanmean(ev_capped)
        welfare["ev_median_all"][r] = np.nanmedian(ev)
        welfare["gain_all"][r] = np.mean(ev > tol) * 100
        welfare["lose_all"][r] = np.mean(ev < -tol) * 100
        for i in range(N_BRACKETS):
            m = agents["income_bracket"] == i
            if np.any(m):
                welfare["ev_mean"][r, i] = np.nanmean(ev_capped[m])
                welfare["ev_median"][r, i] = np.nanmedian(ev[m])
                welfare["gain_pct"][r, i] = np.mean(ev[m] > tol) * 100
                welfare["lose_pct"][r, i] = np.mean(ev[m] < -tol) * 100
        welfare["housed_policy"][r], welfare["rent_share_policy"][r] = housing_by_bracket(agents, houses["rent_charged"])
        for g, m in enumerate(welfare_group_masks(agents, houses)):
            welfare["group_share"][r, g] = np.mean(m) * 100
            if np.any(m):
                welfare["group_gain"][r, g] = np.mean(ev[m] > tol) * 100
                welfare["group_lose"][r, g] = np.mean(ev[m] < -tol) * 100
                welfare["group_ev_median"][r, g] = np.nanmedian(ev[m])

    print("Running baseline Monte Carlo sim:")
    print()
    agents_b, houses_b, mc_b, last_round_b = monte_carlo_sim(n_agents=n_agents,
                                                            n_neighborhoods=n_neighborhoods,
                                                            max_rounds=max_rounds,
                                                            n_runs=n_runs,
                                                            seed=seed,
                                                            plot=False,
                                                            on_run_end=store_baseline,
                                                            **kwargs
                                                            )

    print("Running the policy Monte Carlo sim:")
    print()
    agents_p, houses_p, mc_p, last_round_p = monte_carlo_sim_affordable(n_agents=n_agents,
                                                                        n_neighborhoods=n_neighborhoods,
                                                                        max_rounds=max_rounds,
                                                                        n_runs=n_runs,
                                                                        seed=seed,
                                                                        income_cutoff=income_cutoff,
                                                                        houses_eligible=houses_eligible,
                                                                        lower_price=lower_price,
                                                                        plot=False,
                                                                        on_run_end=compare_policy,
                                                                        **kwargs
                                                                        )
    del base_q, base_c # free the per-agent arrays

    last_round = min(last_round_b, last_round_p)

    # run r of the baseline and run r of the policy start from the same city, so the cleanest test is the
    # paired difference at the final round (most of the city-to-city noise cancels out)
    for stat, label in POLICY_METRICS:
        diff = paired_difference(mc_b, mc_p, stat, confidence)
        if diff is not None:
            print(f"{label} at the final round, policy - baseline: {diff[0]:+.4f} ({confidence}% CI {diff[1]:+.4f} to {diff[2]:+.4f})")

    if plot:
        plot_policy_comparison(mc_b, mc_p, last_round, POLICY_METRICS, confidence, band)
        plot_policy_bracket_happiness(mc_b, mc_p, income_cutoff, confidence, band)

    report_welfare(welfare, income_cutoff, confidence, plot)
    return mc_b, mc_p, last_round, welfare

"--------------------------------------------- welfare: print and plot ----------------------------------------------"
def report_welfare(welfare, income_cutoff = 2, confidence = 95, plot = True):
    med, mlo, mhi = runs_mean_ci(welfare["ev_median_all"], confidence)
    ev, lo, hi = runs_mean_ci(welfare["ev_all"], confidence)
    print()
    print("Welfare (equivalent variation, % of income; + = the policy is worth a rent cut of that size):")
    print(f"  median agent:                   {med:+.3f}% ({confidence}% CI {mlo:+.3f} to {mhi:+.3f})")
    print(f"  equal-weighted mean (capped):   {ev:+.3f}% ({confidence}% CI {lo:+.3f} to {hi:+.3f})")
    print(f"  better off: {np.nanmean(welfare['gain_all']):.1f}% of agents, worse off: {np.nanmean(welfare['lose_all']):.1f}%")
    print()
    import pandas as pd
    print()
    m = lambda k: np.nanmean(welfare[k], axis = 0)
    by_bracket = pd.DataFrame({
        "Median EV (% of income)": m("ev_median"),
        "Mean EV, capped (% of income)": runs_mean_ci(welfare["ev_mean"], confidence)[0],
        "Better off (%)": m("gain_pct"),
        "Worse off (%)": m("lose_pct"),
        "Housed, baseline (%)": m("housed_base"),
        "Housed, policy (%)": m("housed_policy"),
        "Rent share, baseline (%)": m("rent_share_base"),
        "Rent share, policy (%)": m("rent_share_policy"),
    }, index = pd.Index(np.arange(N_BRACKETS), name = "Income bracket")).round(2)
    print("Welfare by income bracket (mean across runs; mean EV capped at +-delta of income per agent):")
    _show_table(by_bracket)

    print()
    by_group = pd.DataFrame({
        "Share of agents (%)": m("group_share"),
        "Better off (%)": m("group_gain"),
        "Worse off (%)": m("group_lose"),
        "Median EV (% of income)": m("group_ev_median"),
    }, index = pd.Index(WELFARE_GROUPS, name = "Where the agent ends up under the policy")).round(2)
    print("Welfare by where agents end up under the policy (mean across runs):")
    _show_table(by_group)
    welfare["by_bracket"] = by_bracket # kept for export, e.g. welfare["by_group"].to_csv(...)
    welfare["by_group"] = by_group

    if plot:
        plot_welfare(welfare, income_cutoff, confidence)
