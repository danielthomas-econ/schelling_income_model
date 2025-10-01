from .agents import *
from .bidding import *
from .common import *
from .houses import *
from .stats import *
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# full disclosure:
# i had no idea how to write the plot_segregation_grid or the create_segregation_animation functions
# so i vibe coded them with Claude
# i understand how it works though

"---------------------------- to visualize the segregation with a heatmap of avg income ----------------------------"
def plot_segregation_grid(stats, last_round, save_path=None):
    # Determine grid layout based on number of rounds
    num_rounds = last_round + 1
    
    # Calculate subplot grid dimensions (roughly square)
    ncols = int(np.ceil(np.sqrt(num_rounds)))
    nrows = int(np.ceil(num_rounds / ncols))
    
    # Create figure with appropriate size and spacing
    fig = plt.figure(figsize=(ncols*2.5, nrows*2.5 + 1.5))
    
    # Create gridspec with space for title and colorbar
    gs = fig.add_gridspec(nrows, ncols, 
                          left=0.05, right=0.95, 
                          top=0.92, bottom=0.08,
                          hspace=0.3, wspace=0.2)
    
    # Get global income statistics for color scale
    all_incomes = stats["avg_income"]["income"][:num_rounds].flatten()
    
    # Set center of colormap to Round 0's city-wide average
    round_0_incomes = stats["avg_income"]["income"][0]
    vcenter = np.nanmean(round_0_incomes)
    
    # Set vmin to 0 (fully red) and make vmax symmetric
    vmin = 0
    vmax = 2 * vcenter  # This makes vcenter the midpoint between 0 and vmax
    
    # Create a diverging colormap (red for poor, green for rich)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', 
              '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('income', colors, N=256)
    
    # Plot each round
    axes = []
    for round_num in range(num_rounds):
        row = round_num // ncols
        col = round_num % ncols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        # Extract income data for this round
        income_data = stats["avg_income"]["income"][round_num]
        neighborhoods = stats["avg_income"]["neighborhood"][round_num]
        
        # Reshape into 10x10 grid
        grid = np.full((10, 10), np.nan)
        for i in range(len(neighborhoods)):
            nb = int(neighborhoods[i])
            income = income_data[i]
            row_idx = nb // 10
            col_idx = nb % 10
            grid[row_idx, col_idx] = income
        
        # Plot heatmap
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, 
                       interpolation='nearest', aspect='equal')
        
        # Formatting
        ax.set_title(f'Round {round_num}', fontsize=9, fontweight='bold', pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Overall title at the top
    fig.suptitle('Income Segregation Dynamics Over Time', 
                 fontsize=18, fontweight='bold', y=0.97)
    
    # Add colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Average Neighborhood Income (₹)', 
                   fontsize=11, fontweight='bold', labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    
    # Format colorbar labels with comma separators, no scientific notation
    from matplotlib.ticker import FuncFormatter
    def format_rupees(x, pos):
        return f'₹{int(x):,}'
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(format_rupees))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
"---------------------------------------- creates a gif of the visualization ----------------------------------------"
def create_segregation_animation(stats, last_round, save_path='segregation_animation.gif'):
    """
    Creates an animated GIF showing segregation dynamics.
    Requires: pip install imageio
    """
    import imageio
    import os
    
    # Create temporary directory for frames
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    num_rounds = last_round + 1
    
    # Get global color scale
    all_incomes = stats["avg_income"]["income"][:num_rounds].flatten()
    vmin = np.nanmin(all_incomes)
    vmax = np.nanmax(all_incomes)
    
    # Create a diverging colormap (red for poor, green for rich)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', 
              '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('income', colors, N=256)
    
    frames = []
    
    for round_num in range(num_rounds):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extract and reshape data
        income_data = stats["avg_income"]["income"][round_num]
        neighborhoods = stats["avg_income"]["neighborhood"][round_num]
        
        grid = np.full((10, 10), np.nan)
        for i in range(len(neighborhoods)):
            nb = neighborhoods[i]
            income = income_data[i]
            row = nb // 10
            col = nb % 10
            grid[row, col] = income
        
        # Plot
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, 
                       interpolation='nearest', aspect='equal')
        
        ax.set_title(f'Round {round_num}/{last_round}', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Avg Income (₹)', fontsize=12, fontweight='bold')
        
        # Save frame
        frame_path = f'{temp_dir}/frame_{round_num:03d}.png'
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        frames.append(imageio.imread(frame_path))
        plt.close()
    
    # Create GIF
    imageio.mimsave(save_path, frames, duration=0.5, loop=0)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Animation saved to {save_path}")


"-------------------------------------- run the sim max_rounds number of times --------------------------------------"
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
    agents = generate_agents(n_agents)
    agents["theta"] = np.random.uniform(θ_min, θ_max, n_agents)
    houses = initialize_houses(agents)
    houses["value"] = np.full(n_agents, starting_house_price)

    n_neighborhoods = np.max(agents["neighborhood"])+1
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
        print(f"Homelessness: {(np.sum(agents["neighborhood"]==-1)*100)/n_agents:.3f}%")
        print()

        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        if count > 0:
            stats["num_bids"][count] = np.count_nonzero(bids)
            stats["winning_bids"][count] = num_winners

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
                if np.sum(stats["churn"][count-convergence_bound:count]) == 0: # churn for the past `convergence bound` rounds has been zero
                    last_round = count-1
                    break
        else:
            if count >= max_rounds: # we use max_rounds if converge flag is false
                last_round = count-1 
                break

    return agents, houses, stats, last_round

"------------------------------------- run a monte carlo sim to reduce variance -------------------------------------"
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

"---------------- debug functions: gives us the time of each process to identify any bloat in the sim ---------------"
def sim_one_round_debug(n_agents = N_AGENTS,
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