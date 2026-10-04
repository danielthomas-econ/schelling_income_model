from .common import *
from .agents import *
from .houses import *
from .bidding import *
from .stats import *
from .policy import *
import time

"---------------- debug functions: gives us the time of each process to identify any bloat in the sim ---------------"
# these mirror run_round (sim.py) and run_round_affordable (policy.py) step by step, printing how long each step takes

class _Timer:
    # tiny helper: t.lap("label") prints the time since the last lap
    def __init__(self):
        self.t = time.time()
    def lap(self, label):
        now = time.time()
        print(f"{label}: {now - self.t:.4f} secs")
        self.t = now

"------------------------------------------- one timed round, baseline ----------------------------------------------"
def _run_round_timed(agents, houses, happiness_percent, delta, nonmarket_quality, temperature,
                     preference = PREFERENCE, homophily_window = HOMOPHILY_WINDOW):
    t = _Timer()
    proportions, q_nm = neighborhood_quality(agents, nonmarket_quality, preference, homophily_window)
    agents = check_happiness(agents, proportions, happiness_percent)
    t.lap("Check happiness")

    stay = stay_values(agents, proportions, q_nm, delta)
    evict_priced_out(agents, houses, check_priced_out(agents, houses["value"], stay))
    t.lap("Evict priced out")

    bids, neighborhoods_chosen = place_bid(agents, proportions, q_nm, houses["value"],
                                           rents_market = bidding_rents(houses),
                                           available_market = vacant_neighborhoods(houses),
                                           delta = delta, temperature = temperature)
    t.lap("Bidding process")
    floor = reservation_rent(agents)
    agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen, stay, floor)
    t.lap("House allocation")
    houses = update_prices(houses, cutoff_bids, floor)
    t.lap("Update prices")
    return agents, houses, bids, num_winners

"-------------------------------------------- one timed round, policy -----------------------------------------------"
def _run_round_affordable_timed(agents, houses, happiness_percent, delta, nonmarket_quality, temperature, lower_price,
                                preference = PREFERENCE, homophily_window = HOMOPHILY_WINDOW):
    t = _Timer()
    proportions, q_nm = neighborhood_quality(agents, nonmarket_quality, preference, homophily_window)
    agents = check_happiness(agents, proportions, happiness_percent)
    t.lap("Check happiness")

    stay = stay_values(agents, proportions, q_nm, delta)
    evict_priced_out(agents, houses, check_priced_out(agents, houses["rent_charged"], stay))
    t.lap("Evict priced out")

    rents_market = bidding_rents(houses)
    rents_eligible = np.where(vacant_neighborhoods(houses, low_rent_only = True),
                              bidding_rents(houses, low_rent = True), rents_market)
    bids, neighborhoods_chosen = place_bid(agents, proportions, q_nm, houses["rent_charged"],
                                           rents_market = rents_market,
                                           available_market = vacant_neighborhoods(houses, market_only = True),
                                           rents_eligible = rents_eligible,
                                           available_eligible = vacant_neighborhoods(houses),
                                           delta = delta, temperature = temperature)
    t.lap("Bidding process")
    floor = reservation_rent(agents)
    agents, houses, cutoff_bids, num_winners = allocate_houses_affordable(agents, houses, bids, neighborhoods_chosen,
                                                                          stay, floor)
    t.lap("House allocation")
    houses = update_prices_affordable(houses, cutoff_bids, floor, lower_price = lower_price)
    t.lap("Update prices")
    agents = update_rent_paid_affordable(agents, houses)
    t.lap("Update rent paid")
    return agents, houses, bids, num_winners

"-------------------------------------------------- baseline sim ----------------------------------------------------"
def sim_one_round_debug(n_agents = N_AGENTS,
                        n_neighborhoods = N_NEIGHBORHOODS,
                        max_rounds = 100,
                        happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                        starting_house_price = STARTING_HOUSE_PRICE,
                        delta = DELTA,
                        nonmarket_quality = NONMARKET_QUALITY,
                        temperature = CHOICE_TEMPERATURE,
                        theta_min = THETA_MIN,
                        theta_max = THETA_MAX,
                        preference = PREFERENCE,
                        homophily_window = HOMOPHILY_WINDOW,
                        converge = False,
                        convergence_bound = 5, # will the sim end if we have churn convergence?
                        seed = None):
    total_start = time.time()
    t = _Timer()
    set_seed(seed)
    agents, houses = new_population(n_agents, starting_house_price, theta_min, theta_max)
    stats = initialize_stats(num_rounds=max_rounds, num_neighborhoods=n_neighborhoods)
    t.lap("Initialization")
    count = 0 # tracks iterations
    prev_house = None # we initialize previous houses = none since ofc its not defined yet

    while not np.all(agents["happy"]):
        print(f"Round {count}")
        agents, houses, bids, num_winners = _run_round_timed(agents, houses, happiness_percent, delta,
                                                             nonmarket_quality, temperature,
                                                             preference, homophily_window)
        t = _Timer()
        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        stats["num_bids"][count] = np.count_nonzero(bids)
        stats["winning_bids"][count] = num_winners
        t.lap("Stats generation")
        print()

        count += 1
        if should_stop(stats["churn"], count, max_rounds, converge, convergence_bound):
            break
    last_round = count-1
    print(f"Total time taken: {time.time()-total_start:.4f} seconds")
    print()
    return agents, houses, stats, last_round

"---------------------------------------------------- monte carlo ---------------------------------------------------"
def monte_carlo_sim_debug(n_agents = N_AGENTS,
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
                          preference = PREFERENCE,
                          homophily_window = HOMOPHILY_WINDOW,
                          converge = False,
                          convergence_bound = 5, # will the sim end if we have churn convergence?
                          seed = None,
                          redraw_population = True):
    t = _Timer()
    set_seed(seed)
    agents_og, houses_og = new_population(n_agents, starting_house_price, theta_min, theta_max)
    mc_stats = initialize_mc_stats(num_runs = n_runs, num_rounds=max_rounds, num_agents=agents_og.size, num_neighborhoods=n_neighborhoods)
    t.lap("Initial initialization")

    for current_run in range(n_runs):
        print(f"Running run {current_run+1}")
        print()
        t = _Timer()
        if seed is not None:
            set_seed(seed + current_run)
        if redraw_population and current_run > 0:
            agents, houses = new_population(n_agents, starting_house_price, theta_min, theta_max)
        else:
            agents = agents_og.copy()
            houses = houses_og.copy()
        t.lap("Drawing the population")
        count = 0 # tracks iterations
        while not np.all(agents["happy"]):
            print(f"    Round {count}")
            agents, houses, bids, num_winners = _run_round_timed(agents, houses, happiness_percent, delta,
                                                                 nonmarket_quality, temperature,
                                                                 preference, homophily_window)
            t = _Timer()
            mc_stats = get_mc_stats(mc_stats, agents, houses, run_id = current_run, current_round=count)
            mc_stats["num_bids"][current_run, count] = np.count_nonzero(bids)
            mc_stats["winning_bids"][current_run, count] = num_winners
            t.lap("Stats generation")

            count += 1
            if should_stop(mc_stats["churn"][current_run], count, max_rounds, converge, convergence_bound):
                break
        mc_stats["last_round"][current_run] = count-1

    last_round = int(np.max(mc_stats["last_round"]))
    return agents, houses, mc_stats, last_round

"--------------------------------------------- affordable housing policy --------------------------------------------"
def sim_one_round_affordable_debug(n_agents = N_AGENTS,
                                   n_neighborhoods = N_NEIGHBORHOODS,
                                   max_rounds = 100,
                                   happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                                   starting_house_price = STARTING_HOUSE_PRICE,
                                   delta = DELTA,
                                   nonmarket_quality = NONMARKET_QUALITY,
                                   temperature = CHOICE_TEMPERATURE,
                                   theta_min = THETA_MIN,
                                   theta_max = THETA_MAX,
                                   preference = PREFERENCE,
                                   homophily_window = HOMOPHILY_WINDOW,
                                   converge = False,
                                   convergence_bound = 5, # will the sim end if we have churn convergence?
                                   income_cutoff = 2, # agents with this income bracket and below are eligible
                                   houses_eligible = 0.2, # what percent of houses in each neighborhood are 'affordable'
                                   lower_price = 0.6, # low rent homes will cost (actual rent) * (lower_price), 60% by default
                                   seed = None):
    total_start = time.time()
    t = _Timer()
    set_seed(seed)
    agents, houses = new_population_affordable(n_agents, starting_house_price, theta_min, theta_max,
                                               income_cutoff, houses_eligible, lower_price)
    stats = initialize_stats(num_rounds=max_rounds, num_neighborhoods=n_neighborhoods)
    t.lap("Initialization")
    count = 0 # tracks iterations
    prev_house = None # we initialize previous houses = none since ofc its not defined yet

    while not np.all(agents["happy"]):
        print(f"Round {count}")
        agents, houses, bids, num_winners = _run_round_affordable_timed(agents, houses, happiness_percent, delta,
                                                                        nonmarket_quality, temperature, lower_price,
                                                                        preference, homophily_window)
        t = _Timer()
        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        stats["num_bids"][count] = np.count_nonzero(bids)
        stats["winning_bids"][count] = num_winners
        t.lap("Stats generation")
        print()

        count += 1
        if should_stop(stats["churn"], count, max_rounds, converge, convergence_bound):
            break
    last_round = count-1

    print(f"Set aside targeting: {set_aside_targeting(agents, houses)}")
    print(f"Time taken: {time.time()-total_start:.4f} seconds")
    print()
    return agents, houses, stats, last_round
