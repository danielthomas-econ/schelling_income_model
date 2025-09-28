"All the constants that the sim relies on"
import numpy as np

# default population of the sim
N_AGENTS = 1_000_000

N_NEIGHBORHOODS = 100
GRID_SIZE = int(np.sqrt(N_NEIGHBORHOODS)) # side of the square which represents our city

# percentiles tells us how many income brackets do you want?
PERCENTILES = [0,10,20,30,40,50,60,70,80,90,95,99,100]
N_BRACKETS = len(PERCENTILES)-1

# something to consider:
# i deleted the grid variable for now, but will it be needed when we have to visualize the model?
# that might restrict n_neighborhoods to being a perfect square

def allocate_neighborhood(agents, n_neighborhoods = N_NEIGHBORHOODS):
    neighborhoods = np.random.randint(0,n_neighborhoods, agents.size)
    return neighborhoods

# what percent of agents must have >= income?
DEFAULT_HAPPINESS_PERCENT = 0.5

# use as a robustness check: does our result change if the starting house price varies?
STARTING_HOUSE_PRICE = 1_00_000

# housing price update rules
DECAY_RATE = 0.95       # fall in price if supply > demand
MAX_CHANGE = 0.1        # max % change in one round, prevents insane price swings

# agent behavior parameters
BETA = 0.3              # baseline fraction of income WTP
LAMBDA = 1            # marginal WTP for social utility
DELTA = 0.6             # maximum income agent can spend on rent

# theta is the preference for neighborhood quality, we'll uniformly distribute it
THETA_MIN = 0.6
THETA_MAX = 0.8
