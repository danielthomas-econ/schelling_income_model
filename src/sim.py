from .agents import *
from .bidding import *
from .common import *
from .houses import *
from .stats import *
from .plots import plot_run_summary, plot_mc_summary
import time
import numpy as np

"------------------------------------------- draw a fresh city to simulate ------------------------------------------"
def new_population(n_agents = N_AGENTS,
                   starting_house_price = STARTING_HOUSE_PRICE,
                   theta_min = THETA_MIN,
                   theta_max = THETA_MAX):
    agents = generate_agents(n_agents)
    agents["theta"] = np.random.uniform(theta_min, theta_max, n_agents)
    houses = initialize_houses(agents)
    houses["value"] = np.full(n_agents, starting_house_price)
    return agents, houses

"---------------------------------------------- one round of the sim ----------------------------------------------"
# sim_one_round, monte_carlo_sim and the debug versions all used to carry their own copy of this logic
def run_round(agents, houses, happiness_percent = DEFAULT_HAPPINESS_PERCENT, delta = DELTA,
              nonmarket_quality = NONMARKET_QUALITY, temperature = CHOICE_TEMPERATURE,
              preference = PREFERENCE, homophily_window = HOMOPHILY_WINDOW):
    # gets the prpn of similar agents in every neighborhood for every bracket, and the quality of nonmarket housing
    proportions, q_nm = neighborhood_quality(agents, nonmarket_quality, preference, homophily_window)
    agents = check_happiness(agents, proportions, happiness_percent)

    # the most each tenant would pay to stay. tenants whose rent is above it leave for nonmarket housing,
    # and the same values are the tenants' claims in the rent auction
    stay = stay_values(agents, proportions, q_nm, delta)
    evict_priced_out(agents, houses, check_priced_out(agents, houses["value"], stay))

    # the whole bidding and house allocation process
    bids, neighborhoods_chosen = place_bid(agents, proportions, q_nm, houses["value"],
                                           rents_market = bidding_rents(houses),
                                           available_market = vacant_neighborhoods(houses),
                                           delta = delta, temperature = temperature)
    agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen, stay)
    houses = update_prices(houses, cutoff_bids)
    return agents, houses, bids, num_winners

"-------------------------------------- run the sim max_rounds number of times --------------------------------------"
def sim_one_round(n_agents = N_AGENTS,
                  n_neighborhoods = N_NEIGHBORHOODS,
                  max_rounds = 100,
                  happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                  starting_house_price = STARTING_HOUSE_PRICE,
                  delta = DELTA,
                  nonmarket_quality = NONMARKET_QUALITY,
                  temperature = CHOICE_TEMPERATURE,
                  theta_min = THETA_MIN,
                  theta_max = THETA_MAX,
                  preference = PREFERENCE, # "status" or "homophily", see common.py
                  homophily_window = HOMOPHILY_WINDOW,
                  converge = False,
                  convergence_bound = 5, # will the sim end if we have churn convergence?
                  seed = None, # set an int to make the run reproducible
                  plot = True):
    # initialization
    start = time.time()
    set_seed(seed)
    agents, houses = new_population(n_agents, starting_house_price, theta_min, theta_max)

    stats = initialize_stats(num_rounds=max_rounds, num_neighborhoods=n_neighborhoods)
    count = 0 # tracks iterations
    prev_house = None # we initialize previous houses = none since ofc its not defined yet

    # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
    while not np.all(agents["happy"]):
        print(f"Round {count}")
        agents, houses, bids, num_winners = run_round(agents, houses, happiness_percent, delta, nonmarket_quality,
                                                      temperature, preference, homophily_window)
        print(f"Happiness: {(np.sum(agents["happy"])*100)/n_agents:.3f}%")
        print()

        # log the data
        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        stats["num_bids"][count] = np.count_nonzero(bids)
        stats["winning_bids"][count] = num_winners

        count += 1
        if should_stop(stats["churn"], count, max_rounds, converge, convergence_bound):
            break
    last_round = count-1 # FIX: also defined now if the loop ends because everyone is happy

    if plot:
        plot_run_summary(stats, agents, last_round)

    end = time.time()
    print(f"Time taken: {end-start:.4f} seconds")
    return agents, houses, stats, last_round

"------------------------------------- run a monte carlo sim to reduce variance -------------------------------------"
def monte_carlo_sim(n_agents = N_AGENTS,
                    n_neighborhoods = N_NEIGHBORHOODS,
                    max_rounds = 100,
                    n_runs = 30,
                    happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                    starting_house_price = STARTING_HOUSE_PRICE,
                    delta = DELTA,
                    nonmarket_quality = NONMARKET_QUALITY,
                    temperature = CHOICE_TEMPERATURE,
                    theta_min = THETA_MIN,
                    theta_max = THETA_MAX,
                    preference = PREFERENCE, # "status" or "homophily", see common.py
                    homophily_window = HOMOPHILY_WINDOW,
                    converge = False,
                    convergence_bound = 5, # will the sim end if we have churn convergence?
                    seed = None, # run r uses seed + r, so the whole MC is reproducible
                    redraw_population = True, # draw new incomes and starting neighborhoods every run
                    plot = True,
                    on_run_end = None): # optional function(run_id, agents, houses), called with each run's final state
    # FIX: every run used to start from the same agents_og/houses_og copy, so the only thing that varied across runs
    # was the tiebreaker in place_bid. that made the CIs far too narrow. now each run draws a new city by default
    # set redraw_population = False to get the old behavior back (useful to isolate dynamic noise from sampling noise)
    start = time.time()
    set_seed(seed)
    agents_og, houses_og = new_population(n_agents, starting_house_price, theta_min, theta_max)

    mc_stats = initialize_mc_stats(num_runs = n_runs, num_rounds=max_rounds, num_agents=agents_og.size, num_neighborhoods=n_neighborhoods)

    final_happiness_by_bracket = mc_stats["bracket_happiness"] # (n_runs, n_brackets), filled in at the end of each run

    for current_run in range(n_runs):
        print(f"Running run {current_run+1}")
        print()
        if seed is not None:
            set_seed(seed + current_run)
        if redraw_population and current_run > 0:
            agents, houses = new_population(n_agents, starting_house_price, theta_min, theta_max)
        else:
            agents = agents_og.copy()
            houses = houses_og.copy()
        count = 0 # tracks iterations
        # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
        while not np.all(agents["happy"]):
            agents, houses, bids, num_winners = run_round(agents, houses, happiness_percent, delta, nonmarket_quality,
                                                          temperature, preference, homophily_window)

            # log the data
            mc_stats = get_mc_stats(mc_stats, agents, houses, run_id = current_run, current_round=count)
            mc_stats["num_bids"][current_run, count] = np.count_nonzero(bids)
            mc_stats["winning_bids"][current_run, count] = num_winners

            count += 1
            # FIX: this used to slice mc_stats["churn"] along the runs axis instead of the rounds axis,
            # and had no max_rounds cap when converge = True (which would crash past max_rounds)
            if should_stop(mc_stats["churn"][current_run], count, max_rounds, converge, convergence_bound):
                break

        mc_stats["last_round"][current_run] = count-1

        # get data on happiness by income bracket at the end of the run
        for i in range(N_BRACKETS):
            mask = agents["income_bracket"] == i
            total_i = np.sum(mask)
            if total_i > 0:
                final_happiness_by_bracket[current_run, i] = np.sum(agents["happy"][mask]) * 100 / total_i

        # lets evaluate_policy read each run's final state (used for the welfare comparison)
        if on_run_end is not None:
            on_run_end(current_run, agents, houses)

    # runs can end at different rounds when converge = True, so plot up to the longest one
    last_round = int(np.max(mc_stats["last_round"]))

    if plot:
        plot_mc_summary(mc_stats, last_round)

    end = time.time()
    print(f"Total time taken: {end-start:.4f} secs")
    return agents, houses, mc_stats, last_round

"------------------------- look at the impact changing parameters has on different variables ------------------------"
def parameter_sweep(n_agents = 10_000, # results are mostly robust to popln size, so using 1m agents here would just slow us down
                    n_neighborhoods = N_NEIGHBORHOODS,
                    max_rounds = 100,
                    n_runs = 5, # using 5 MC runs to save on load
                    happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                    starting_house_price = STARTING_HOUSE_PRICE,
                    delta = DELTA,
                    nonmarket_quality = NONMARKET_QUALITY,
                    temperature = CHOICE_TEMPERATURE,
                    theta_min = THETA_MIN,
                    theta_max = THETA_MAX,
                    preference = PREFERENCE, # "status" or "homophily", see common.py
                    homophily_window = HOMOPHILY_WINDOW,
                    params = None, # the parameters we want to evaluate here
                    sensitivity = 100, # how many values of the parameter do we evaluate for the sweep? higher -> more values evaluated
                    converge = False,
                    convergence_bound = 5,
                    seed = None): # same seed at every param value => common random numbers, so differences come from the param

    # ------------------------------------- just making sure params has valid entries ------------------------------------ #
    if params == None:
        params = [] # creates a fresh list each call, safer
    if params is not None and not isinstance(params, list): # enforce type
        raise TypeError("params must be a list")
    valid = ["delta", "nonmarket_quality", "temperature", "theta_min", "theta_max"]
    for i in params:
        if i not in valid:
            raise ValueError(f"element {i} not in valid params: {valid}")
    # ------------------------------------------------ param checking done ----------------------------------------------- #

    # store results here (obviously)
    all_results = {} # for each param
    # we also include std dev tracking to plot the results with CIs
    results_dtype = np.dtype([
        ("param_value", np.float32), # value of the param for which we get results
        ("avg_value", np.float32),
        ("avg_value_std", np.float32),
        ("nonmarket_housing", np.float32), # percentage
        ("nonmarket_housing_std", np.float32),
        ("vacancies", np.float32), # percentage
        ("vacancies_std", np.float32),
        ("happiness", np.float32), # percentage
        ("happiness_std", np.float32),
        ("gini", np.float32), # percentage
        ("gini_std", np.float32),
        ("theil", np.float32), # >= 0, no upper bound
        ("theil_std", np.float32),
        ("theil_within", np.float32),
        ("theil_within_std", np.float32),
        ("theil_between", np.float32),
        ("theil_between_std", np.float32),
        ("churn", np.float32),
        ("churn_std", np.float32),
    ])

    # set the ranges
    for param_name in params:
        if param_name == "delta":
            param_values = np.linspace(0.3, 0.9, sensitivity) # high sensitivity -> lower gaps in linspace
        elif param_name == "nonmarket_quality":
            param_values = np.linspace(0.05, 0.95, sensitivity)
        elif param_name == "temperature":
            param_values = np.linspace(0.02, 0.5, sensitivity)
        elif param_name == "theta_min":
            param_values = np.linspace(0.05, theta_max-0.05, sensitivity) # bounded above by theta_max
        elif param_name == "theta_max":
            param_values = np.linspace(theta_min + 0.05, 0.95, sensitivity) # bounded below

        # initialize a new results array for this param
        # len param_values -> one entry for each column for each param value
        results = np.zeros(len(param_values), dtype = results_dtype)
        for idx, param_val in enumerate(param_values):
            print(f"Running sim at {param_name} = {param_val:.4f}, {idx+1}/{len(param_values)}")

            # we use all the kwargs passed into the function as is, just changing the value of the current param being swept
            kwargs = {
                "n_agents": n_agents,
                "n_neighborhoods": n_neighborhoods,
                "max_rounds": max_rounds,
                "n_runs": n_runs,
                "happiness_percent": happiness_percent,
                "starting_house_price": starting_house_price,
                "delta": delta,
                "nonmarket_quality": nonmarket_quality,
                "temperature": temperature,
                "theta_min": theta_min,
                "theta_max": theta_max,
                "preference": preference,
                "homophily_window": homophily_window,
                "converge": converge,
                "convergence_bound": convergence_bound,
                "seed": seed,
                "plot": False,
            }
            kwargs[param_name] = param_val # the only update to kwargs each round

            # run the mc sim with all the given arguments + the param value for this iteration
            agents, houses, mc_stats, last_round = monte_carlo_sim(**kwargs)
            results["param_value"][idx] = param_val # store the param value corresponding to the results

            # FIX: take each run's value at its own last round (runs can end at different rounds when converge = True)
            run_ids = np.arange(n_runs)
            run_ends = mc_stats["last_round"]

            # looping the print statement instead of manually doing it, thanks Claude
            metrics_final = ["avg_value", "nonmarket_housing", "vacancies", "happiness", "gini", "theil", "theil_within", "theil_between"]
            for metric in metrics_final:
                final_values = mc_stats[metric][run_ids, run_ends]
                results[metric][idx] = np.mean(final_values)
                results[f"{metric}_std"][idx] = np.std(final_values)

            # we want an avg over all rounds for churn, so we have to treat it a bit differently
            # unlike the other params where the last round info is enough
            avg_churn_per_run = np.array([np.mean(mc_stats["churn"][r, 1:run_ends[r]+1]) if run_ends[r] > 0 else 0.0
                                          for r in run_ids])
            results["churn"][idx] = np.mean(avg_churn_per_run)
            results["churn_std"][idx] = np.std(avg_churn_per_run)

            print(f"Happiness: {results["happiness"][idx]:3f}% ± {results["happiness_std"][idx]:.2f}")
            print(f"Nonmarket housing: {results["nonmarket_housing"][idx]:.3f}% ± {results["nonmarket_housing_std"][idx]:.2f}")
            print()
            # clears up memory, esp since mc_stats can be quite big
            del agents, houses, mc_stats

        all_results[param_name] = results # save the results for a given parameter in its corresponding key

    return all_results
