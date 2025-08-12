import numpy as np
import scipy.stats as st
from numba import njit, jit
from houses import *

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

"-------------------------------------------- generate the agents finally -------------------------------------------"
def generate_agents(n_agents=n_agents, mean_income=mean_income, log_std=log_std, cutoffs=cutoffs):
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
    return incomes, brackets, locations, neighborhood

"---------------------------------- list of arrays that represent each neighborhood ---------------------------------"
@njit
def neighborhood_lists(neighborhood):
    neighborhood_list = [] # a list where the ith index represents an array for neighborhood i, and jth index of that array is jth resident of i
    for i in range(100):
        residents = np.where(neighborhood == i)[0] # consists of residents of neighborhood i
        neighborhood_list.append(residents)
    return neighborhood_list

"----- agent wants {happiness_percent}% of people in his neighborhood to be of the same income bracket or higher ----"
@njit
def check_happiness(array, happiness_percent = 0.5):
    array = np.asarray(array)
    n = array.size
    happiness_threshold = n * happiness_percent * 100

    # freq basically becomes an array counting frequency of each income bracket
    freq = np.bincount(array, minlength=12)
    # np.cumsum adds up freq and gives us the number of agents with <= cumsum[v]
    # n - np.cumsum flips that and gives no. of agents with > cumsum[v]
    # + freq allows equality so it becomes >= cumsum[v], our condition
    cumsum = n - np.cumsum(freq) + freq
    return (cumsum[array] * 100 >= happiness_threshold) # returns True/False for each agent

"--------------------------------- creates a list of arrays with happiness booleans ---------------------------------"
def is_happy(neighborhood, incomes, cutoffs = cutoffs, happiness_percent = 0.5):
    neighborhood_list = neighborhood_lists(neighborhood)
    happiness_list = []
    for n in neighborhood_list: # for each neighborhood in the sim
        n_income = incomes[n]
        brackets = np.searchsorted(cutoffs, n_income) - 1 # lists each member of the neighborhood by their income bracket instead of agent index no
        # list of arrays that contain whether agent is happy or not
        # same format as neighborhood, so the indices correspond
        happiness_list.append(check_happiness(brackets))
    return happiness_list 

"--------------------------------- initializes an array to check if agent has a home --------------------------------"
@njit
def initialize_tenancy_list(neighborhood):
    is_tenant = [np.zeros_like(array) for array in neighborhood]
    return is_tenant


    
