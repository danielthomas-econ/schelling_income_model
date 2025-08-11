from agents import *
from houses import *
import numpy as np
from numba import njit, jit

"------------------------------------------- cobb-douglas utility function ------------------------------------------"
def utility(θ, start, end, brackets_start, brackets_end):
    # given one starting neighborhood, ending neighborhood and all the income brackets (one array)
    # return an array of utility values for the agents in starting neighborhood
    
    return


"--------------------------------------------------- bidding logic --------------------------------------------------"
def place_bid(neighborhood, incomes,
               β = 0.3, # base fraction of income agent is wtp
               λ = 0.2, # marginal WTP for 1 unit of social utility U
               δ = 0.6 # max cap on affordability, so that bids dont take up entire agent income
               ):
    is_tenant = initialize_tenancy_list(neighborhood)
    houses, vacanies = initialize_houses(neighborhood)