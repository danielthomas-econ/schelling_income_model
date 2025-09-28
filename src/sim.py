from .agents import *
from .bidding import *
from .common import *
from .houses import *
from .stats import *
import time

import numpy as np
import matplotlib.pyplot as plt

"---------------------------- to visualize the segregation with a heatmap of avg income ----------------------------"
def plot_sim(agents, houses, num_nbs, fig_size=(4,4)):
    plt.figure(figsize=fig_size)
    gridsize = int(np.sqrt(num_nbs)) # 100 nbs will give us a square of side 10
    plt.axhline(xmin=1, xmax=gridsize-1)
    plt.axvline(ymin=1, ymax=gridsize-1)


def sim_one_round(n_agents = N_AGENTS,
                  max_rounds = 100,
                  happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                  starting_house_price = STARTING_HOUSE_PRICE, 
                  β = BETA, 
                  λ = LAMBDA, 
                  δ = DELTA, 
                  θ_min = THETA_MIN,
                  θ_max = THETA_MAX,
                  converge = False,
                  convergence_bound = 5): # will the sim end if we have churn convergence?
    # initialization
    start = time.time()
    agents = generate_agents(n_agents)
    agents["theta"] = np.random.uniform(θ_min, θ_max, n_agents)
    houses = initialize_houses(agents)
    houses["value"] = np.full(n_agents, starting_house_price)

    n_neighborhoods = np.max(agents["neighborhood"])+1
    stats = initialize_stats(num_rounds=max_rounds, num_neighborhoods=n_neighborhoods)
    end = time.time()
    print(f"Initialization: {end-start:.4f} secs")
    count = 0 # tracks iterations
    prev_house = None # we initialize previous houses = none since ofc its not defined yet

    # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
    while not np.all(agents["happy"]):
        print(f"Round {count}")
        # gets the prpn of agents >= income brackets for all brackets and calculates happiness
        start = time.time()
        freq, total = get_freq_and_total(agents)
        proportions = get_proportion(freq, total)
        agents = check_happiness(agents, proportions, happiness_percent)
        end = time.time()
        print(f"Check happiness: {end-start:.4f} secs")
        #print(f"Happiness: {(np.sum(agents["happy"])*100)/n_agents:.3f}%")
        #print(f"Homelessness: {(np.sum(agents["neighborhood"]==-1)*100)/n_agents:.3f}%")
        #print()

        start = time.time()
        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        if count > 0:
            stats["num_bids"][count] = np.count_nonzero(bids)
            stats["winning_bids"][count] = num_winners
        end = time.time()
        print(f"Stats generation: {end-start:.4f} secs")        

        start = time.time()
        priced_out_mask = check_priced_out(agents, houses, proportions, β, λ, δ) # look at who can no longer afford their home
        evict_priced_out(agents, houses, priced_out_mask) # start by evicting them, so they can also participate in this round of bidding
        end = time.time()
        print(f"Evict priced out: {end-start:.4f} secs")

        # the whole bidding and house allocation process
        current_rents = get_current_rents(houses)
        start = time.time()
        utilities = get_utilities(agents, proportions, current_rents)
        end = time.time()
        print(f"Utility calculation: {end-start:.4f} secs")
        start = time.time()
        bids, neighborhoods_chosen = place_bid(agents, utilities, β, λ, δ)
        end = time.time()
        print(f"Bidding process: {end-start:.4f} secs")
        start = time.time()
        agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen)
        end = time.time()
        print(f"House allocation: {end-start:.4f} secs")
        start = time.time()
        houses = update_prices(agents, houses, neighborhoods_chosen, cutoff_bids, β = β)
        end = time.time()
        print(f"Update prices: {end-start:.4f} secs")

        count += 1
        if converge == True:
            if count - convergence_bound >= 0:
                if np.sum(stats["churn"][count-convergence_bound:count]) == 0: # churn for the past `convergence bound` rounds has been zero
                    last_round = count-1
                    break
        else:
            if count >= max_rounds: # we use max_rounds if converge flag is false
                last_round = count-1 
                break

    return agents, houses, stats, last_round


def monte_carlo_sim(n_agents = N_AGENTS,
                    max_rounds = 100,
                    n_runs = 30,
                    happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                    starting_house_price = STARTING_HOUSE_PRICE, 
                    β = BETA, 
                    λ = LAMBDA, 
                    δ = DELTA, 
                    θ_min = THETA_MIN,
                    θ_max = THETA_MAX,
                    converge = False,
                    convergence_bound = 5): # will the sim end if we have churn convergence?
    # initialization
    agents_og = generate_agents(n_agents)
    agents_og["theta"] = np.random.uniform(θ_min, θ_max, n_agents)
    houses_og = initialize_houses(agents_og)
    houses_og["value"] = np.full(n_agents, starting_house_price)
    

    n_neighborhoods = np.max(agents_og["neighborhood"])+1
    mc_stats = initialize_mc_stats(num_runs = n_runs, num_rounds=max_rounds, num_agents=agents_og.size, num_neighborhoods=n_neighborhoods)

    for current_run in range(n_runs):
        print(f"Running run {current_run+1}")
        print()
        agents = agents_og.copy()
        houses = houses_og.copy()
        count = 0 # tracks iterations
        # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
        while not np.all(agents["happy"]):
            print(f"    Round {count}")
            # gets the prpn of agents >= income brackets for all brackets and calculates happiness
            freq, total = get_freq_and_total(agents)
            proportions = get_proportion(freq, total)
            agents = check_happiness(agents, proportions, happiness_percent)

            mc_stats = get_mc_stats(mc_stats, agents, houses, run_id = current_run, current_round=count)
            if count > 0:
                mc_stats["num_bids"][current_run, count] = np.count_nonzero(bids)
                mc_stats["winning_bids"][current_run, count] = num_winners

            priced_out_mask = check_priced_out(agents, houses, proportions, β, λ, δ) # look at who can no longer afford their home
            evict_priced_out(agents, houses, priced_out_mask) # start by evicting them, so they can also participate in this round of bidding

            # the whole bidding and house allocation process
            current_rents = get_current_rents(houses)
            utilities = get_utilities(agents, proportions, current_rents)
            bids, neighborhoods_chosen = place_bid(agents, utilities, β, λ, δ)
            agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen)
            houses = update_prices(agents, houses, neighborhoods_chosen, cutoff_bids, β = β)

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

    return agents, houses, mc_stats, last_round

def monte_carlo_sim_debug(n_agents = N_AGENTS,
                            max_rounds = 100,
                            n_runs = 30,
                            happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                            starting_house_price = STARTING_HOUSE_PRICE, 
                            β = BETA, 
                            λ = LAMBDA, 
                            δ = DELTA, 
                            θ_min = THETA_MIN,
                            θ_max = THETA_MAX,
                            converge = False,
                            convergence_bound = 5): # will the sim end if we have churn convergence?
    # initialization
    start = time.time()
    agents_og = generate_agents(n_agents)
    agents_og["theta"] = np.random.uniform(θ_min, θ_max, n_agents)
    houses_og = initialize_houses(agents_og)
    houses_og["value"] = np.full(n_agents, starting_house_price)
    

    n_neighborhoods = np.max(agents_og["neighborhood"])+1
    mc_stats = initialize_mc_stats(num_runs = n_runs, num_rounds=max_rounds, num_agents=agents_og.size, num_neighborhoods=n_neighborhoods)
    end = time.time()
    print(f"Intial initalization time: {end-start:.4f} secs")

    for current_run in range(n_runs):
        print(f"Running run {current_run+1}")
        print()
        agents = agents_og.copy()
        houses = houses_og.copy()
        count = 0 # tracks iterations
        # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
        while not np.all(agents["happy"]):
            print(f"    Round {count}")
            # gets the prpn of agents >= income brackets for all brackets and calculates happiness
            start = time.time()
            freq, total = get_freq_and_total(agents)
            proportions = get_proportion(freq, total)
            agents = check_happiness(agents, proportions, happiness_percent)
            end = time.time()
            print(f"Check happiness: {end-start:.4f} secs")

            start = time.time()
            mc_stats = get_mc_stats(mc_stats, agents, houses, run_id = current_run, current_round=count)
            if count > 0:
                mc_stats["num_bids"][current_run, count] = np.count_nonzero(bids)
                mc_stats["winning_bids"][current_run, count] = num_winners
            end = time.time()
            print(f"Stats generation: {end-start:.4f} secs")         

            start = time.time()
            priced_out_mask = check_priced_out(agents, houses, proportions, β, λ, δ) # look at who can no longer afford their home
            evict_priced_out(agents, houses, priced_out_mask) # start by evicting them, so they can also participate in this round of bidding
            end = time.time()
            print(f"Evict priced out: {end-start:.4f} secs")

            # the whole bidding and house allocation process
            current_rents = get_current_rents(houses)
            start = time.time()
            utilities = get_utilities(agents, proportions, current_rents)
            end = time.time()
            print(f"Utility calculation: {end-start:.4f} secs")
            start = time.time()
            bids, neighborhoods_chosen = place_bid(agents, utilities, β, λ, δ)
            end = time.time()
            print(f"Bidding process: {end-start:.4f} secs")
            start = time.time()
            agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen)
            end = time.time()
            print(f"Allocating houses: {end-start:.4f} secs")
            start = time.time()
            houses = update_prices(agents, houses, neighborhoods_chosen, cutoff_bids, β = β)
            end = time.time()
            print(f"Update prices: {end-start:.4f} secs")

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

    return agents, houses, mc_stats, last_round