"All the constants that the sim relies on"

# population of the sim
N_AGENTS = 1_000_000

GRID_SIZE = 10 # the length of the square's side
N_NEIGHBORHOODS = GRID_SIZE**2 # since the grid is a square, we have side^2 neighborhoods

# what percent of agents must have >= income?
DEFAULT_HAPPINESS_PERCENT = 0.5

# housing price update rules
DECAY_RATE = 0.95       # fall in price if supply > demand
MAX_CHANGE = 0.2         # max % change in one round, prevents insane price swings

# agent behavior parameters
BETA = 0.3               # baseline fraction of income WTP
LAMBDA = 0.2             # marginal WTP for social utility
DELTA = 0.6              # maximum income agent can spend on rent

# percentiles tells us how many income brackets do you want?
PERCENTILES = [0,10,20,30,40,50,60,70,80,90,95,99,100]

# theta is the preference for neighborhood quality, we'll uniformly distribute it
THETA_MIN = 0.6
THETA_MAX = 0.8
