import numpy as np
import scipy.stats as st
from numba import njit, jit, prange
from .houses import * #.houses => relative import, prevents import errors in plots.ipynb

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

cutoffs = find_income_brackets()

"----- agent wants {happiness_percent}% of people in his neighborhood to be of the same income bracket or higher ----"
@njit(parallel = True)
def check_happiness(agents, happiness_percent = 0.5):
    n = agents.size
    income_brackets = agents["income_bracket"]
    neighborhoods = agents["neighborhood"]

    # 100 -> no. of neighborhoods (0-99), 12 -> no. of income brackets (0-11)
    freq_per_neighborhood = np.zeros((100,12), dtype = np.int32)
    total_per_neighborhood = np.zeros((100), dtype = np.int32)

    # how many agents of each income_bracket live in each neighborhood (and total agents in a neighborhood)
    for i in range(n):
        nb = neighborhoods[i]
        ib = income_brackets[i]
        freq_per_neighborhood[nb,ib] += 1
        total_per_neighborhood[nb] += 1 

    cumsum_per_neighborhood = np.zeros((100,12), dtype = np.int32)
    # parallelizing with prange since every nb works on a different row
    for nb in prange(100):
        running_sum = 0
        # iterate over brackets backwards to get >= bracket count
        for ib in range(11,-1,-1):
            running_sum += freq_per_neighborhood[nb, ib]
            cumsum_per_neighborhood[nb,ib] = running_sum

    # computes happiness for each agent
    for i in prange(n):
        nb = neighborhoods[i]
        ib = income_brackets[i]
        condition = cumsum_per_neighborhood[nb,ib] / total_per_neighborhood[nb]
        agents["happy"][i] = (condition >= happiness_percent)
        
    return agents

"-------------------------------------------- generate the agents finally -------------------------------------------"
def generate_agents(n_agents=n_agents, mean_income=mean_income, log_std=log_std, cutoffs=cutoffs):
    # gen a structured array to store everything
    # low level memory optimization
    # just for 9 columns, it has bought down memory consumption by ~82% vs the list of arrays structure
    agent_dtype = np.dtype([
        ("id", np.int32),
        ("income", np.float64),
        ("income_bracket", np.uint8),
        ("neighborhood", np.uint8),
        ("happy", np.bool_),
        ("bid", np.float64),
        ("rent_paid", np.float64), # no need for checking tenancy, rent_paid = 0 => not a tenant
        ("priced_out_threshold", np.float64), # use the priced out formula
        ("theta", np.float32),
    ])
    
    mu = np.log(mean_income) - 0.5 * (log_std ** 2) # math to make 2.5L the actual mean of the lognormal distr
    
    incomes = np.random.lognormal(mean=mu, sigma=log_std, size=n_agents)
    brackets = np.searchsorted(cutoffs, incomes) - 1 # sorts the incomes into the cutoffs list
    
    locations = np.random.uniform(0, 10, size=(n_agents, 2))
    # get_neighborhood_num expects a single location, so vectorize it here:
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
    agents["bid"] = np.zeros(n_agents)
    agents["rent_paid"] = np.zeros(n_agents)
    agents["priced_out_threshold"] = np.zeros(n_agents)
    agents["theta"] = np.random.uniform(0.6,0.8, n_agents)
    return agents
