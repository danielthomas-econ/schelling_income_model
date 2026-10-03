"All the constants that the sim relies on"
import numpy as np
from numba import njit

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

# note: this no longer affects results. every neighborhood starts empty, so round 0's auction sets the first rents
STARTING_HOUSE_PRICE = 1_00_000

# housing price update rules
DECAY_RATE = 0.95       # fall in price if supply > demand
MAX_CHANGE = 0.1        # max % change in one round, prevents insane price swings

# ------------------------------------------- agent preferences and budgets ------------------------------------------ #
# utility of living in neighborhood k at rent r:  U = q^theta * c^(1-theta)
#   q = share of k's residents at or above the agent's income bracket
#   c = (DELTA*y - r) / (DELTA*y): the share of the agent's housing budget left after rent
# DELTA is the share of income available for housing; the other (1 - DELTA) is committed to non-housing necessities.
# rents above DELTA*y give c = 0 => U = 0, so no agent ever pays more than DELTA of their income in rent
DELTA = 0.65

# theta is the weight on neighborhood quality vs leftover budget, drawn uniformly per agent
# the ends are trimmed: theta near 0 means an agent doesn't value neighbors at all, and since a home's only value in
# the model is its neighborhood, they'd always prefer nonmarket housing
THETA_MIN = 0.1
THETA_MAX = 0.9

# nonmarket housing (the outside option): no rent (c = 1), and its quality for an agent in bracket b is
#   q_nm(b) = NONMARKET_QUALITY * s(b),   s(b) = city-wide share of agents at or above bracket b
# i.e. nonmarket housing is like living in a randomly mixed neighborhood, discounted by NONMARKET_QUALITY.
# why not a single q_nm for everyone: q is measured relative to the agent's own bracket, so for the top 1% a mixed
# neighborhood has q ~ 0.01. a constant q_nm of, say, 0.2 would make every mixed neighborhood worse than nonmarket
# housing for brackets 9-11; they would never bid, never cluster, and stay in nonmarket housing forever
NONMARKET_QUALITY = 0.1

# do agents in nonmarket housing still count towards the composition of the neighborhood they were evicted from?
# False => only agents with a market home are residents. this drives neighborhood composition (proportions)
# and the gini/theil/avg income stats. set to True as a robustness check
COUNT_NONMARKET = False

# ------------------------------------------ how agents choose where to bid ------------------------------------------ #
# unhappy agents choose ONE neighborhood among those that (1) have a vacant home they could rent, (2) they prefer to
# nonmarket housing at the current rent, and (3) if they already have a home, they strictly prefer to it.
# the choice is logit: P(k) ∝ exp((U_k / U_best - 1) / CHOICE_TEMPERATURE), so the best option is the most likely
# but close alternatives get picked too. this stops a whole income bracket from piling onto the same neighborhood.
# smaller => closer to always picking the best one (must be > 0). report results for a couple of values
CHOICE_TEMPERATURE = 0.1

"---------------------------------------------- the cobb-douglas model ----------------------------------------------"
# these two functions are the whole preference model. bidding, evictions and the rent auction all use them
@njit(cache = True)
def utility(income, theta, q, rent, delta):
    budget = delta * income
    c = (budget - rent) / budget # share of the housing budget left after rent
    if c <= 0.0 or q <= 0.0:
        return 0.0
    return q ** theta * c ** (1.0 - theta)

# bid-rent: the highest rent at which a home with quality q is still at least as good as nonmarket housing
# solves q^theta * c^(1-theta) = q_nm^theta (nonmarket housing has c = 1) for the rent:
#   r* = delta * y * (1 - (q_nm / q)^(theta / (1 - theta)))
# r* is what an agent bids to move in, and the most they'll pay to stay: rent > r* => they leave for nonmarket housing
@njit(cache = True)
def bid_rent(income, theta, q, q_nm, delta):
    if q <= q_nm:
        return 0.0
    return delta * income * (1.0 - (q_nm / q) ** (theta / (1.0 - theta)))

def resident_mask(agents, count_nonmarket = COUNT_NONMARKET):
    # boolean mask of agents who count as residents of their neighborhood
    if count_nonmarket:
        return np.ones(agents.size, dtype=np.bool_)
    return ~agents["nonmarket_housing"]

"--------------------------------------------- when does a run stop? ---------------------------------------------"
# one shared stopping rule for every sim loop. count = number of rounds completed so far
# max_rounds is always a hard cap (the stats arrays are only max_rounds long, so going past it would crash)
# with converge = True, we also stop once churn has been zero for `convergence_bound` rounds in a row
def should_stop(churn_history, count, max_rounds, converge, convergence_bound):
    if count >= max_rounds:
        return True
    if converge and count >= convergence_bound:
        if np.sum(churn_history[count-convergence_bound:count]) == 0:
            return True
    return False

"------------------------------------------------ reproducible seeds ------------------------------------------------"
# numba keeps its own random state, separate from numpy's, so np.random.seed in python doesn't reach
# the np.random calls inside jitted functions (eg the neighborhood choice in place_bid). we have to seed both
@njit(cache = True)
def _seed_numba(seed):
    np.random.seed(seed)

def set_seed(seed):
    if seed is None:
        return
    # FIX: numpy and numba both use the Mersenne Twister, so seeding both with the SAME number gives them the SAME
    # stream. the uniforms that create incomes then reappear as the uniforms agents use to choose a neighborhood, and
    # round 0 comes out perfectly sorted by income. derive two independent seeds instead
    numpy_seed, numba_seed = np.random.SeedSequence(seed).generate_state(2)
    np.random.seed(int(numpy_seed))
    _seed_numba(int(numba_seed))
