from .agents import *
from .bidding import *
from .common import *
from .houses import *
from .stats import *
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import imageio
import os
import shutil

"-------------------------------------- run the sim max_rounds number of times --------------------------------------"
def sim_one_round(n_agents = N_AGENTS,
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
                  convergence_bound = 5): # will the sim end if we have churn convergence?
    # initialization
    start = time.time()
    agents = generate_agents(n_agents)
    agents["theta"] = np.random.uniform(theta_min, theta_max, n_agents)
    houses = initialize_houses(agents)
    houses["value"] = np.full(n_agents, starting_house_price)

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

        priced_out_mask = check_priced_out(agents, houses, proportions, beta, gamma, delta) # look at who can no longer afford their home
        evict_priced_out(agents, houses, priced_out_mask) # start by evicting them, so they can also participate in this round of bidding

        # the whole bidding and house allocation process
        current_rents = get_current_rents(houses)
        utilities = get_utilities(agents, proportions, current_rents)
        bids, neighborhoods_chosen = place_bid(agents, utilities, beta, gamma, delta)
        agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen)
        houses = update_prices(agents, houses, neighborhoods_chosen, cutoff_bids, beta = beta)

        # log the data
        stats, prev_house = get_stats(stats, agents, houses, current_round=count, prev_house=prev_house)
        if count > 0:
            stats["num_bids"][count] = np.count_nonzero(bids)
            stats["winning_bids"][count] = num_winners

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
    plt.title("Happiness over time")
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
    plt.title("Happiness by income bracket")
    plt.xlabel("Income bracket")
    plt.ylabel("Happiness (%)")
    plt.show()

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
                    beta = BETA, 
                    gamma = GAMMA, 
                    delta = DELTA, 
                    theta_min = THETA_MIN,
                    theta_max = THETA_MAX,
                    converge = False,
                    convergence_bound = 5): # will the sim end if we have churn convergence?
    start = time.time()
    # initialization
    agents_og = generate_agents(n_agents)
    agents_og["theta"] = np.random.uniform(theta_min, theta_max, n_agents)
    houses_og = initialize_houses(agents_og)
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
        # ideally we wanna run this sim until all agents are happy, but thats very unlikely to ever happen
        while not np.all(agents["happy"]):
            # gets the prpn of agents >= income brackets for all brackets and calculates happiness
            freq, total = get_freq_and_total(agents)
            proportions = get_proportion(freq, total)
            agents = check_happiness(agents, proportions, happiness_percent)

            priced_out_mask = check_priced_out(agents, houses, proportions, beta, gamma, delta) # look at who can no longer afford their home
            evict_priced_out(agents, houses, priced_out_mask) # start by evicting them, so they can also participate in this round of bidding

            # the whole bidding and house allocation process
            current_rents = get_current_rents(houses)
            utilities = get_utilities(agents, proportions, current_rents)
            bids, neighborhoods_chosen = place_bid(agents, utilities, beta, gamma, delta)
            agents, houses, cutoff_bids, num_winners = allocate_houses(agents, houses, bids, neighborhoods_chosen)
            houses = update_prices(agents, houses, neighborhoods_chosen, cutoff_bids, beta = beta)

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

"------------------------- look at the impact changing parameters has on different variables ------------------------"
def parameter_sweep(n_agents = 10_000, # results are mostly robust to popln size, so using 1m agents here would just slow us down
                    n_neighborhoods = N_NEIGHBORHOODS,
                    max_rounds = 100,
                    n_runs = 5, # using 5 MC runs to save on load
                    happiness_percent = DEFAULT_HAPPINESS_PERCENT,
                    starting_house_price = STARTING_HOUSE_PRICE, 
                    beta = BETA, 
                    gamma = GAMMA, 
                    delta = DELTA, 
                    theta_min = THETA_MIN,
                    theta_max = THETA_MAX,
                    params = None, # the parameters we want to evaluate here
                    sensitivity = 100, # how many values of the parameter do we evaluate for the sweep? higher -> more values evaluated
                    converge = False,
                    convergence_bound = 5):
    
    # ------------------------------------- just making sure params has valid entries ------------------------------------ #
    if params == None:
        params = [] # creates a fresh list each call, safer
    if params is not None and not isinstance(params, list): # enforce type
        raise TypeError("params must be a list")
    valid = ["beta", "gamma", "delta", "theta_min", "theta_max"]
    for i in params:
        if i not in valid:
            raise ValueError(f"element {i} not in valid params: ['beta', 'gamma', 'delta', 'theta_min', 'theta_max']")
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
        ("dissimilarity", np.float32),
        ("dissimilarity_std", np.float32),
        ("churn", np.float32),
        ("churn_std", np.float32),
    ])

    # set the ranges
    for param_name in params:
        if param_name in ["beta", "gamma", "delta"]: # all have same range 0.05-0.95
            param_values = np.linspace(0.05,0.95, sensitivity) # high sensitivity -> lower gaps in linspace
        elif param_name == "theta_min":
            param_values = np.linspace(0.1, theta_max-0.05, sensitivity) # bounded above by theta_max
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
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "theta_min": theta_min,
                "theta_max": theta_max,
                "converge": converge,
                "convergence_bound": convergence_bound
            }
            kwargs[param_name] = param_val # the only update to kwargs each round

            # run the mc sim with all the given arguments + the param value for this iteration
            agents, houses, mc_stats, last_round = monte_carlo_sim(**kwargs)
            results["param_value"][idx] = param_val # store the param value corresponding to the results

            # looping the print statement instead of manually doing it, thanks Claude
            metrics_final = ["avg_value", "nonmarket_housing", "vacancies", "happiness", "gini", "theil", "theil_within", "theil_between"]
            for metric in metrics_final:
                final_values = mc_stats[metric][:, last_round]
                results[metric][idx] = np.mean(final_values)
                results[f"{metric}_std"][idx] = np.std(final_values)

            # we want an avg over all rounds for churn, so we have to treat it a bit differently
            # unlike the other params where the last round info is enough
            avg_churn_per_run = np.mean(mc_stats["churn"][:, 1:last_round+1], axis=1) # axis = 1 => sum along columns, gives us avg for each run instead of each round
            results["churn"][idx] = np.mean(avg_churn_per_run)
            results["churn_std"][idx] = np.std(avg_churn_per_run)

            print(f"Happiness: {results["happiness"][idx]:3f}% ± {results["happiness_std"][idx]:.2f}")
            print(f"Nonmarket housing: {results["nonmarket_housing"][idx]:.3f}% ± {results["nonmarket_housing_std"][idx]:.2f}")
            print()
            # clears up memory, esp since mc_stats can be quite big
            del agents, houses, mc_stats

        all_results[param_name] = results # save the results for a given parameter in its corresponding key

    return all_results

"-------------------------------------- plot the results of the parameter sweep -------------------------------------"
def plot_parameter_sweep(all_results, save_path=None):
    fig, axes = plt.subplots(3,2,figsize=(15,12))
    axes = axes.flatten() # gives us 1d indexing

    # all the metrics we'll plot
    metrics = [
        ("happiness", "Happiness (%)"),
        ("nonmarket_housing", "Nonmarket Housing (%)"),
        ("gini", "Gini Index"),
        ("theil_between", "Theil Between"),
        ("churn", "Average Churn (%)"),
        ("avg_value", "Average House Value (₹)")
    ]

    # much better way to plot all at once instead of doing it all individually, once again thanks to Claude
    for param_name, results in all_results.items():
        param_values = results["param_value"]
        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx]
            mean_vals = results[metric]
            std_vals = results[f"{metric}_std"]

            ax.plot(param_values, mean_vals, label = param_name)
            # plot CIs
            ax.fill_between(param_values, mean_vals - std_vals, mean_vals + std_vals, alpha = 0.5)

            ax.set_xlabel(f"Parameter value")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs {param_name}")
            ax.legend()
            ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches = "tight")
        print(f"Saved the parameter sweep plot to {save_path}")
    else:
        plt.show()


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
    shutil.rmtree(temp_dir)
    
    print(f"Animation saved to {save_path}")