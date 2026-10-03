from .common import *
import numpy as np
import scipy.interpolate as si
from numba import njit, jit, prange
import pandas as pd
import os

np.set_printoptions(suppress=True) # stops scientific numbers

"------------------------- get agents income from a cdf of indian income based on CMIE data -------------------------"
def get_incomes(agents):
    here = os.path.dirname(__file__) # current path of agents.py (/src)
    # goes one level up to fetch the csv
    filepath = os.path.join(here, "..", "data", "income_quantile_delhi.csv") # now we use delhi specific data instead of india level data
    filepath = os.path.abspath(filepath) # gets an actual absolute path

    file = pd.read_csv(filepath) # hard coded path so the notebooks wont face an issue
    quantiles = file["quantile_value"].values
    income = file["income"].values

    log_income = np.log(income)
    # s = 5 gives us a smoother fit than using pchip interpolator
    # this makes the income plots look less jagged and spiky
    spline = si.UnivariateSpline(x=quantiles, y=log_income, s=5)
    uniform = np.random.uniform(0,1,agents.size)
    agent_incomes = np.exp(spline(uniform)) # pass the uniform array through the quantile to get agent incomes as the output

    return agent_incomes
"------------------------------------------ computing income brackets once ------------------------------------------"
def find_income_brackets(agents, percentiles = PERCENTILES):
    # FIX: the cutoffs used to come from a PCHIP fit of the CSV, while incomes are drawn from a *smoothed* spline
    # (s = 5) of the same CSV. the two curves don't match, so the brackets weren't true deciles
    # (in testing, bracket 0 held ~12% of agents and the top-1% bracket could end up empty)
    # now the cutoffs are percentiles of the simulated incomes themselves, so bracket 0 is exactly the bottom 10%, etc
    cutoffs = np.percentile(agents["income"], percentiles).astype(np.float64)
    cutoffs[0] = 0
    cutoffs[-1] = np.inf # ensures lowest cutoff is zero and that there is no highest cutoff

    brackets = np.searchsorted(cutoffs, agents["income"]) - 1
    return brackets

"----------------- get frequency of each bracket in a neighborhood + total agents in a neighborhood -----------------"
# writing this outside check_happiness so i can use np.bincount here while still parallelizing with numba later
# this is so much faster
def get_freq_and_total(agents, count_nonmarket = COUNT_NONMARKET):
    # only residents count towards a neighborhood's composition (see COUNT_NONMARKET in common.py)
    residents = resident_mask(agents, count_nonmarket)
    neighborhoods = agents["neighborhood"][residents]
    income_brackets = agents["income_bracket"][residents]
    freq = np.zeros((N_NEIGHBORHOODS, N_BRACKETS), dtype = np.int32)

    # takes a (nb,ib) pair as a coordinate and increments freq[nb,ib] by 1
    # directly fills in the freq values, we're done with updating this now
    np.add.at(freq, (neighborhoods, income_brackets), 1)

    # maybe im a bit too obsessed with declaring datatypes now :)
    # axis = 1 => sum over columns to get no. of agents in each neighborhood
    total = freq.sum(axis=1).astype(np.int32)

    return freq, total

"------------------------- quality of a randomly mixed neighborhood, for each income bracket ------------------------"
# s(b) = city-wide share of agents at or above bracket b. this is what q would be in a perfectly mixed neighborhood
# used for (1) empty neighborhoods, including every neighborhood in round 0, and (2) nonmarket housing quality
def city_quality(agents, n_brackets = N_BRACKETS):
    counts = np.bincount(agents["income_bracket"], minlength = n_brackets)[:n_brackets]
    at_or_above = np.cumsum(counts[::-1])[::-1] # running sum from the top bracket down
    return (at_or_above / agents.size).astype(np.float64)

# quality of nonmarket housing for each bracket: a randomly mixed neighborhood, discounted (see NONMARKET_QUALITY)
def nonmarket_quality_by_bracket(agents, nonmarket_quality = NONMARKET_QUALITY):
    return nonmarket_quality * city_quality(agents)

"------------ run this outside of check_happiness so i can reuse the logic later for utility evaluations ------------"
# FIX: the decorator was missing its '@', so this line did nothing and the function ran as plain (slow) python
@njit(parallel = True, cache = True)
def get_proportion(freq, total, empty_quality):
    # precomputes an array to check what proportion in neighborhood j has >= income bracket i
    proportions = np.zeros((N_NEIGHBORHOODS,N_BRACKETS), dtype = np.float64)
    # parallelizing with prange since every nb works on a different row
    for nb in prange(N_NEIGHBORHOODS):
        if total[nb] == 0:
            # nobody lives there, so its composition is unknown: agents expect a random mix (city_quality)
            # this covers round 0, when every neighborhood is empty. with q = 0 instead, nobody would ever bid
            for ib in range(N_BRACKETS):
                proportions[nb,ib] = empty_quality[ib]
            continue
        # code for the >= income bracket logic
        running_sum = 0
        # iterate over brackets backwards to get >= bracket count
        for ib in range(N_BRACKETS-1, -1,-1):
            running_sum += freq[nb, ib]
            proportions[nb, ib] = running_sum / total[nb]
    return proportions

"----- agent wants {happiness_percent}% of people in his neighborhood to be of the same income bracket or higher ----"
@jit(parallel = True, cache = True)
def check_happiness(agents, proportions, happiness_percent = DEFAULT_HAPPINESS_PERCENT):
    n = agents.size
    income_brackets = agents["income_bracket"]
    neighborhoods = agents["neighborhood"]

    # computes happiness for each agent
    for i in prange(n):
        nb = neighborhoods[i]
        ib = income_brackets[i]
        if agents["nonmarket_housing"][i] == True:
            agents["happy"][i] = False # people living in nonmarket housing arent happy
        else:
            agents["happy"][i] = (proportions[nb, ib] >= happiness_percent)

    return agents

"-------------------------------------------- generate the agents finally -------------------------------------------"
def generate_agents(n_agents = N_AGENTS):
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
