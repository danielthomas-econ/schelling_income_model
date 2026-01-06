from numba import jit, njit, prange
import numpy as np
import time
from .houses import *
from .common import *
from .agents import *
from .stats import *
from .bidding import *
from .sim import *

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
@jit(parallel = True, cache = True)
def assign_low_rent_house(houses, houses_eligible = 0.2, # low rent homes will cost (actual rent) * (lower_price), 60% by default
                          ):
    neighborhoods = houses["neighborhood"]

    for i in prange(max(neighborhoods)+1):
        mask = np.where(houses["neighborhood"] == i)[0] # lets us work with indices instead of typical boolean array
        k = int(houses_eligible * len(mask)) # how many houses are eligible?
        houses["low_rent"][mask[:k]] = True # we set the first k houses in our neighborhood to be low rent ones
    return houses

"-------------------------------------- pricing system under affordable housing -------------------------------------"
def update_prices_affordable(agents, houses, neighborhood_chosen, cutoff_bids,
                            decay_rate = DECAY_RATE, # fall in price if supply > demand
                            max_change = MAX_CHANGE, # maximum % change in price in one round
                            beta = BETA, # same beta as before, used to calculate price floors
                            lower_price = 0.6, # low rent homes will cost (actual rent) * (lower_price), 60% by default
                            ):
    
    n_neighborhoods = np.max(houses["neighborhood"]) + 1
    
    for n in prange(n_neighborhoods):
        mask = houses["neighborhood"] == n
        low_rent = mask & (houses["low_rent"] == True) # new mask for low rent houses
        agents_mask = agents["neighborhood"] == n
        old_price = houses["value"][mask] # current prices

        vacancies = np.sum(mask & (houses["tenant"] == -1)) # number of vacant units is the available supply
        demand = np.sum(neighborhood_chosen == n) # demand for neighborhood n
        cutoff = cutoff_bids[n]

        # minimum sustainable price: WTP of the poorest agent when utility is zero (beta * income)
        # MSP is like the greatest lower bound on prices
        # if price < MSP, then markets should move prices upwards since even the poorest agent can easily afford the current rent
        # if price >= MSP, then this is because agents derive some social utility from staying in the neighborhood
        incomes = agents["income"][agents_mask]
        if incomes.size > 0: # avoid bugs where no residents are in the neighborhood
            P_floor = beta * np.min(incomes)
        else:
            P_floor = 0.0

        if demand >= vacancies and cutoff > 0:
            # update the prices of all houses in neighborhood n to be the cutoff price
            new_price = np.full_like(old_price,cutoff)
        else: # cutoff = 0 implies there weren't any bids
            new_price = old_price * decay_rate # slight price decay

        # clip the house prices so that changes aren't too drastic
        clipped = np.clip(new_price, a_min = old_price * (1-max_change), a_max = old_price * (1+max_change))
        houses["value"][mask] = np.maximum(clipped, P_floor) # ensures price >= MSP

        # affordability update
        houses["rent_charged"][mask] = houses["value"][mask] # set it equal to the current market rate
        houses["rent_charged"][low_rent] *= lower_price
    return houses

"--------------------------------- check if an agent can no longer afford their home --------------------------------"
@njit(parallel = True, cache = True)
def check_priced_out_affordable(agents, houses, proportions, beta = BETA, gamma = GAMMA, delta = DELTA):
    n_agents = agents.size
    priced_out_mask = np.zeros(n_agents, dtype=np.bool_)

    for i in prange(n_agents):
        h = agents["house"][i]
        if h == -1:
            continue # homeless agents dont have a home they can be priced out of anyways
        
        nb = agents["neighborhood"][i]
        income = agents["income"][i]
        bracket = agents["income_bracket"][i]
        Q_i = proportions[nb, bracket]
        B_stay = min((beta + gamma*Q_i)*income, delta*income) # priced out logic
        rent = houses["rent_charged"][h] # base it off the rent charged, not market value of the house (small change from original)
        if rent > B_stay:
            priced_out_mask[i] = True
    
    return priced_out_mask

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


"--------------------------------- the affordable housing policy version of the sim ---------------------------------"
def sim_one_round_affordable(n_agents = N_AGENTS,
                  n_neighborhoods = N_NEIGHBORHOODS,
                  max_rounds = 100,
                  happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                  starting_house_price = STARTING_HOUSE_PRICE, 
                  beta = BETA, 
                  gamma = GAMMA, 
                  delta = DELTA, 
                  theta_min = THETA_MIN,
                  theta_max = THETA_MAX,
                  converge = False,
                  convergence_bound = 5, # will the sim end if we have churn convergence?
                  income_cutoff = 2, # agents with this income bracket and below are eligible
                  houses_eligible = 0.2, # what percent of houses in each neighborhood are 'affordable'
                  lower_price = 0.6, # low rent homes will cost (actual rent) * (lower_price), 60% by default
                  ): 
    start = time.time()
    # initialization
    agents = generate_agents_affordable(n_agents)
    agents["theta"] = np.random.uniform(theta_min, theta_max, n_agents)
    houses = initialize_houses_affordable(agents)
    houses["value"] = np.full(n_agents, starting_house_price)

    # assign eligible agents and houses
    agents = check_low_rent_eligibility(agents, income_cutoff)
    houses = assign_low_rent_house(houses, houses_eligible)

    stats = initialize_stats(num_rounds=max_rounds, num_neighborhoods=n_neighborhoods)
    count = 0 # tracks iterations
    prev_house = None # we initialize previous houses = none since ofc its not defined yet

    # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
    while not np.all(agents["happy"]):
        print(f"Round {count}")
        # gets the prpn of agents >= income brackets for all brackets and calculates happiness
        freq, total = get_freq_and_total(agents)
        proportions = get_proportion(freq, total)
        agents = check_happiness(agents, proportions, happiness_percent)
        print(f"Happiness: {(np.sum(agents["happy"])*100)/n_agents:.3f}%")
        print()

        # use the affordable version of check priced out
        priced_out_mask = check_priced_out_affordable(agents, houses, proportions, beta, gamma, delta) # look at who can no longer afford their home
        evict_priced_out(agents, houses, priced_out_mask) # start by evicting them, so they can also participate in this round of bidding

        # the whole bidding and house allocation process
        current_rents = get_current_rents(houses)
        utilities = get_utilities(agents, proportions, current_rents)
        bids, neighborhoods_chosen = place_bid(agents, utilities, beta, gamma, delta)
        agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen)
        houses = update_prices_affordable(agents, houses, neighborhoods_chosen, cutoff_bids, beta = beta, lower_price = lower_price)
        
        # log the data
        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        if count > 0:
            stats["num_bids"][count] = np.count_nonzero(bids)
            stats["winning_bids"][count] = num_winners

        # use the correct way to update rent paid
        agents = update_rent_paid_affordable(agents, houses)

        count += 1
        if converge == True:
            # even with converge = True, you cant exceed max rounds as a safety feature to prevent rare cases of very long sims
            if count >= max_rounds:
                last_round = count-1 
                break
            if count - convergence_bound >= 0:
                if np.sum(stats["churn"][count-convergence_bound:count]) == 0: # churn for the past `convergence bound` rounds has been zero
                    last_round = count-1
                    break
        else:
            if count >= max_rounds: # just using max_rounds criteria, no extra convergence criteria
                last_round = count-1 
                break

    # plot happiness over time
    index = np.arange(0, last_round+1) # our x axis
    plt.plot(index, stats["happiness"][:last_round+1], label = "Happiness")
    plt.legend()
    plt.title("Happiness over time (affordable housing policy)")
    plt.xlabel("Rounds")
    plt.ylabel("Happiness (%)")
    plt.show()

    # plot happiness by income bracket
    max_brackets = np.max(agents["income_bracket"]) + 1 # we must add one to this to account for zero being an income bracket
    index = np.arange(max_brackets)
    happy = np.zeros(max_brackets) 

    # print the output in text too
    for i in range(max_brackets):
        mask = agents["income_bracket"] == i
        happy_ib = np.sum(agents["happy"][mask])
        total_ib = np.size(agents[mask])
        happy[i] = (happy_ib*100)/total_ib # prpn of happy agents
        print(f"Income bracket {i}: {happy_ib}/{total_ib} agents happy, {round(happy[i],3)}%")

    plt.bar(index, happy)
    plt.title("Happiness by income bracket (affordable housing policy)")
    plt.xlabel("Income bracket")
    plt.ylabel("Happiness (%)")
    plt.show()

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
                    beta = BETA, 
                    gamma = GAMMA, 
                    delta = DELTA, 
                    theta_min = THETA_MIN,
                    theta_max = THETA_MAX,
                    converge = False,
                    convergence_bound = 5, # will the sim end if we have churn convergence?
                    income_cutoff = 2, # agents with this income bracket and below are eligible
                    houses_eligible = 0.2, # what percent of houses in each neighborhood are 'affordable'
                    lower_price = 0.6, # low rent homes will cost (actual rent) * (lower_price), 60% by default
                    ): 
    start = time.time()
    # initialization
    agents_og = generate_agents_affordable(n_agents)
    agents_og["theta"] = np.random.uniform(theta_min, theta_max, n_agents)
    houses_og = initialize_houses_affordable(agents_og)
    houses_og["value"] = np.full(n_agents, starting_house_price)
    
    mc_stats = initialize_mc_stats(num_runs = n_runs, num_rounds=max_rounds, num_agents=agents_og.size, num_neighborhoods=n_neighborhoods)

    # for plotting happiness by income bracket
    max_brackets = np.max(agents_og["income_bracket"]) + 1
    final_happiness_by_bracket = np.zeros((n_runs, max_brackets))

    for current_run in range(n_runs):
        print(f"Running run {current_run+1}")
        print()
        agents = agents_og.copy()
        houses = houses_og.copy()
        count = 0 # tracks iterations

        # assign eligible agents and houses
        agents = check_low_rent_eligibility(agents, income_cutoff)
        houses = assign_low_rent_house(houses, houses_eligible)

        # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
        while not np.all(agents["happy"]):
            # gets the prpn of agents >= income brackets for all brackets and calculates happiness
            freq, total = get_freq_and_total(agents)
            proportions = get_proportion(freq, total)
            agents = check_happiness(agents, proportions, happiness_percent)

            priced_out_mask = check_priced_out_affordable(agents, houses, proportions, beta, gamma, delta) # look at who can no longer afford their home
            evict_priced_out(agents, houses, priced_out_mask) # start by evicting them, so they can also participate in this round of bidding

            # the whole bidding and house allocation process
            current_rents = get_current_rents(houses)
            utilities = get_utilities(agents, proportions, current_rents)
            bids, neighborhoods_chosen = place_bid(agents, utilities, beta, gamma, delta)
            agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen)
            houses = update_prices_affordable(agents, houses, neighborhoods_chosen, cutoff_bids, beta = beta, lower_price=lower_price)

            # log the data
            mc_stats = get_mc_stats(mc_stats, agents, houses, run_id = current_run, current_round=count)
            if count > 0:
                mc_stats["num_bids"][current_run, count] = np.count_nonzero(bids)
                mc_stats["winning_bids"][current_run, count] = num_winners

            # get data on happiness by income bracket
            for i in range(max_brackets):
                mask = agents["income_bracket"] == i
                total_i = np.sum(mask)
                happy_i = np.sum(agents["happy"][mask])
                final_happiness_by_bracket[current_run, i] = happy_i * 100 / total_i

            count += 1
            if converge == True:
                if count - convergence_bound >= 0:
                    if np.sum(mc_stats["churn"][count-convergence_bound:count]) == 0: # churn for the past `convergence bound` rounds has been zero
                        last_round = count-1
                        break
            else:
                if count >= max_rounds: # we use max_rounds if converge flag is false
                    last_round = count-1 
                    break

    # plot happiness over time
    index = np.arange(0,last_round+1)
    mean_happiness = np.zeros(last_round+1)
    for t in range(last_round + 1):
        vals = mc_stats["happiness"][:, t]
        vals = vals[vals != 0]  # exclude runs that already terminated
        if vals.size > 0:
            mean_happiness[t] = np.mean(vals)
        else:
            mean_happiness[t] = np.nan
    plt.plot(index, mean_happiness, linewidth=2)
    plt.title("Average Happiness Over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Happiness (%)")
    plt.show()

    # plot happiness by income bracket
    brackets = np.arange(max_brackets)
    mean_final_happiness = np.nanmean(final_happiness_by_bracket)
    happy = np.zeros(max_brackets)

    # print a text output
    for i in range(max_brackets):
        mask = agents["income_bracket"] == i
        happy[i] = np.nanmean(final_happiness_by_bracket[:, i]) # avg pct happy in bracket i
        total_ib = np.size(agents[mask])
        happy_ib = round(happy[i]/100 * total_ib,3) # get a numerical value since happy[i] is a pct value
        print(f"Income bracket {i}: {happy_ib}/{total_ib} agents happy on average, {round(happy[i],3)}%")

    plt.bar(brackets, happy)
    plt.title("Happiness by Income Bracket")
    plt.xlabel("Income bracket")
    plt.ylabel("Happiness (%)")

    end = time.time()
    print(f"Total time taken: {end-start:.4f} secs")
    return agents, houses, mc_stats, last_round

"------------------------------------ compare the policy outcomes to the baseline -----------------------------------"
def evaluate_policy(n_agents=N_AGENTS,
                    n_neighborhoods=N_NEIGHBORHOODS,
                    max_rounds=100,
                    n_runs=30,
                    confidence=95,
                    **kwargs
                    ):

    print("Running baseline Monte Carlo sim:")
    print()
    agents_b, houses_b, mc_b, last_round_b = monte_carlo_sim(n_agents=n_agents,
                                                            n_neighborhoods=n_neighborhoods,
                                                            max_rounds=max_rounds,
                                                            n_runs=n_runs,
                                                            **kwargs
                                                            )

    print("Running the policy Monte Carlo sim:")
    print()
    agents_p, houses_p, mc_p, last_round_p = monte_carlo_sim_affordable(n_agents=n_agents,
                                                                        n_neighborhoods=n_neighborhoods,
                                                                        max_rounds=max_rounds,
                                                                        n_runs=n_runs,
                                                                        **kwargs
                                                                        )

    last_round = min(last_round_b, last_round_p)
    index = np.arange(last_round + 1)

    # same helper function as the plot_mc_stats one
    def get_mean_ci(mc_stats, stat_name):
        # CI calculations
        alpha = (100 - confidence) / 100
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        data = mc_stats[stat_name][:, :last_round + 1] # get the data for the relevant time period

        # get each value for each round
        mean = np.zeros(last_round + 1)
        lower = np.zeros(last_round + 1)
        upper = np.zeros(last_round + 1)

        for t in range(last_round + 1):
            # keep only runs that are alive at round t bcz MC runs are variable in length
            vals = data[:, t]
            vals = vals[vals != 0]  # zero means run already terminated

            if vals.size > 0:
                mean[t] = np.mean(vals)
                lower[t] = np.percentile(vals, lower_percentile)
                upper[t] = np.percentile(vals, upper_percentile)
            else:
                mean[t] = np.nan
                lower[t] = np.nan
                upper[t] = np.nan

        return mean, lower, upper

    # the metrics we're gonna look at to evaluate our policy
    metrics = [
        ("happiness", "Happiness (%)"),
        ("nonmarket_housing", "Nonmarket housing (%)"),
        ("churn", "Churn (%)"),
        ("gini", "Gini"),
        ("theil_between", "Theil (between)")
    ]

    for stat, label in metrics:
        mean_b, low_b, up_b = get_mean_ci(mc_b, stat)
        mean_p, low_p, up_p = get_mean_ci(mc_p, stat)

        plt.figure(figsize=(10, 6))

        plt.plot(index, mean_b, label="Baseline", linewidth=2)
        plt.fill_between(index, low_b, up_b, alpha=0.25)

        plt.plot(index, mean_p, label="Affordable housing policy", linewidth=2)
        plt.fill_between(index, low_p, up_p, alpha=0.25)

        plt.title(f"{label}: Baseline vs Policy")
        plt.xlabel("Rounds")
        plt.ylabel(label)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    return