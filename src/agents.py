import numpy as np
import scipy.stats as st
from numba import njit, jit, prange

# way to globally access these variables
mean_income = 460_000 # 4.6 lakhs, avg income in delhi
log_std = 1.2 # lognormal sigma parameter, 1.2 seems to work good in testing
n_agents = 1_000_000 # 1 mil popln
np.set_printoptions(suppress=True) # stops scientific numbers

"------------------------------------------ computing income brackets once ------------------------------------------"
def find_income_brackets(mean_income = mean_income, log_std = log_std, percentiles = [0,10,20,30,40,50,60,70,80,90,95,99,100]):
    # if X is lognormal(mu, sigma), then pth percentile is
    # x_p = exp(mu + sigma*z_p), where z_p is phi inverse(p)
    mu = np.log(mean_income) - 0.5 * (log_std**2)
    probs = np.array(percentiles)/100 # converts probs into a pdf so we can use it to get phi inverse
    
    cutoffs = np.empty(probs.shape) 
    # explictly handle 0 and 1
    cutoffs[0] = 0 # 0th percentile is zero rupees
    cutoffs[-1] = np.inf # 100th percentile is infinite income
    mask = (probs > 0) & (probs < 1) # use a boolean filter to avoid 0 and 1 in further calculations
    
    z = st.norm.ppf(probs[mask]) # takes phi inv exluding 0 and 1 with the mask
    cutoffs[mask] = np.exp(mu + log_std * z) # the percentile formula mentioned at the start
    return cutoffs

"----------------- get frequency of each bracket in a neighborhood + total agents in a neighborhood -----------------"
# writing this outside check_happiness so i can use np.bincount here while still parallelizing with numba later
# this is so much faster
def get_freq_and_total(agents):
    neighborhoods = agents["neighborhood"]
    income_brackets = agents["income_bracket"]

    # 100 -> no. of neighborhoods (0-99), 12 -> no. of income brackets (0-11)
    freq = np.zeros((100,12), dtype = np.int32)
    # avoid homeless people living in neighborhood -1
    valid = neighborhoods != -1
    # takes a (nb,ib) pair as a coordinate and increments freq[nb,ib] by 1
    # directly fills in the freq values, we're done with updating this now
    if valid.any():
        np.add.at(freq, (neighborhoods[valid], income_brackets[valid]), 1)

    # maybe im a bit too obsessed with declaring datatypes now :)
    # axis = 1 => sum over columns to get no. of agents in each neighborhood
    total = freq.sum(axis=1).astype(np.int32)

    return freq, total

"------------ run this outside of check_happiness so i can reuse the logic later for utility evaluations ------------"
@njit(parallel = True, cache = True)
def get_proportion(freq, total):
    # precomputes an array to check what proportion in neighborhood j has >= income bracket i
    proportions = np.zeros((100,12), dtype = np.float32)
    # parallelizing with prange since every nb works on a different row
    for nb in prange(100): 
        if total[nb] == 0: # if no agents live there: avoids division by zero errors
            for ib in range(12):
                proportions[nb,ib] = 0.0 # no agents => proportions = 0 for every income bracket
            continue
        running_sum = 0
        # iterate over brackets backwards to get >= bracket count
        for ib in range(11,-1,-1):
            running_sum += freq[nb, ib]
            proportions[nb, ib] = running_sum / total[nb]
    return proportions

"----- agent wants {happiness_percent}% of people in his neighborhood to be of the same income bracket or higher ----"
@njit(parallel = True, cache = True)
def check_happiness(agents, proportions, happiness_percent = 0.5): 
    n = agents.size
    income_brackets = agents["income_bracket"]
    neighborhoods = agents["neighborhood"]
    
    # computes happiness for each agent
    for i in prange(n):
        nb = neighborhoods[i]
        ib = income_brackets[i]
        if nb == -1:
            agents["happy"][i] = False # homeless people arent happy
        else:
            agents["happy"][i] = (proportions[nb, ib] >= happiness_percent)
        
    return agents

"-------------------------------------------- generate the agents finally -------------------------------------------"
def generate_agents(n_agents=n_agents, mean_income=mean_income, log_std=log_std):
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
        ("rent_paid", np.float64), # no need for checking tenancy, rent_paid = 0 => not a tenant
        ("theta", np.float32), # numba doesnt like float16, so we stick to float32 here
    ])
    
    mu = np.log(mean_income) - 0.5 * (log_std ** 2) # math to make mean_income the actual mean of the lognormal distr
    cutoffs = find_income_brackets()
    incomes = np.random.lognormal(mean=mu, sigma=log_std, size=n_agents)
    brackets = np.searchsorted(cutoffs, incomes) - 1 # sorts the incomes into the cutoffs list
    
    locations = np.random.uniform(0, 10, size=(n_agents, 2))
    # floor the coords and calculate neighborhood numbers
    floored = np.floor(locations).astype(int)
    x_coords = floored[:, 0]
    y_coords = floored[:, 1]
    # we have 100 1x1 neighborhoods in a 10x10 grid
    # start from 0-9 on the bottom row, all the way to 90-99 on the top row
    # so every cell number when labelled like this ends with the floor of the x coord and begins with the floor of the y coord
    neighborhood = y_coords * 10 + x_coords    

    # initialize agents
    agents = np.zeros(n_agents, dtype=agent_dtype)

    agents["id"] = np.arange(n_agents) 
    agents["income"] = incomes
    agents["income_bracket"] = brackets
    agents["neighborhood"] = neighborhood
    agents["happy"] = False # initially everyone is depressed :(
    agents["house"] = np.full(n_agents,-1)
    agents["rent_paid"] = np.zeros(n_agents)
    agents["theta"] = np.random.uniform(0.6,0.8, n_agents)
    return agents
