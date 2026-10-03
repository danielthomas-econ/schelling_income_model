"All the plotting and printed summaries. the sim modules only compute, and call these when plot = True"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # to remove 1e6 base from the x axis on plots
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.stats import t as student_t
import imageio
import os
import shutil
from .common import resident_mask, N_BRACKETS
from .stats import mc_mean_ci, runs_mean_ci

"------------------------------------ summary at the end of a single sim run ----------------------------------------"
# used by sim_one_round and sim_one_round_affordable
def plot_run_summary(stats, agents, last_round, title_suffix = ""):
    # plot happiness over time
    index = np.arange(0, last_round+1) # our x axis
    plt.plot(index, stats["happiness"][:last_round+1], label = "Happiness")
    plt.legend()
    plt.title(f"Happiness over time{title_suffix}")
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
    plt.title(f"Happiness by income bracket{title_suffix}")
    plt.xlabel("Income bracket")
    plt.ylabel("Happiness (%)")
    plt.show()

"-------------------------------------- summary at the end of a monte carlo sim -------------------------------------"
# used by monte_carlo_sim and monte_carlo_sim_affordable
def plot_mc_summary(mc_stats, last_round, title_suffix = ""):
    # plot happiness over time
    index = np.arange(0,last_round+1)
    mean_happiness, _, _ = mc_mean_ci(mc_stats, "happiness", last_round)
    plt.plot(index, mean_happiness, linewidth=2)
    plt.title(f"Average Happiness Over Time{title_suffix}")
    plt.xlabel("Rounds")
    plt.ylabel("Happiness (%)")
    plt.show()

    # plot happiness by income bracket
    brackets = np.arange(N_BRACKETS)
    happy = np.nanmean(mc_stats["bracket_happiness"], axis=0) # avg pct happy in each bracket across runs

    # print a text output
    for i in range(N_BRACKETS):
        print(f"Income bracket {i}: {round(happy[i],3)}% of agents happy on average")

    plt.bar(brackets, happy)
    plt.title(f"Happiness by Income Bracket{title_suffix}")
    plt.xlabel("Income bracket")
    plt.ylabel("Happiness (%)")
    plt.show()

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

"--------------------------------------------- plot graphs of each stat ---------------------------------------------"
def plot_stats(stats, agents, houses, n_neighborhoods, last_round):
    index = np.arange(0, last_round+1) # will always be our x axis

    # happiness
    print(f"Final happiness: {stats["happiness"][last_round]:.3f}%")
    plt.plot(index, stats["happiness"][:last_round+1], label = "Average happiness")
    plt.legend()
    plt.title("Average happiness over time")
    plt.xlabel("Rounds")
    plt.ylabel("Happiness")
    plt.show()

    # agents living in nonmarket housing
    print(f"Final agents in nonmarket housing: {stats['nonmarket_housing'][last_round]:.3f}%")
    plt.plot(index, stats["nonmarket_housing"][:last_round+1])
    plt.title("Agents living in nonmarket housing over time")
    plt.xlabel("Rounds")
    plt.ylabel("Percent of agents")
    plt.show()

    # house value
    print(f"Final average house value: {stats["avg_value"][last_round]:.3f}")
    plt.plot(index, stats["avg_value"][:last_round+1], label = "Average house value")
    plt.legend()
    plt.title("Average house value over time")
    plt.xlabel("Rounds")
    plt.ylabel("House value")
    plt.show()

    # churn
    # FIX: average only over rounds that actually ran (round 0 has no churn by definition)
    print(f"Avg churn: {np.mean(stats["churn"][1:last_round+1]):.3f}%")
    plt.plot(index, stats["churn"][:last_round+1], label = "Churn")
    plt.legend()
    plt.title("Churn per round")
    plt.xlabel("Rounds")
    plt.ylabel("Percent of movers")
    plt.show()

    # gini
    print(f"Final Gini: {stats["gini"][last_round]:.3f}")
    plt.plot(index, stats["gini"][:last_round+1], label = "gini")
    plt.legend()
    plt.title("Avg Gini across neighborhood per round")
    plt.xlabel("Rounds")
    plt.ylabel("Gini index value")
    plt.show()
    # avg gini falls => more homogeneity within each neighborhood because of segregation

    # theil indices
    print(f"Final Theil: {stats["theil"][last_round]:.3f}")
    print(f"Final Theil within: {stats["theil_within"][last_round]:.3f}")
    print(f"Final Theil between: {stats["theil_between"][last_round]:.3f}")
    print(f"Final global Theil: {stats["theil_within"][last_round]+stats["theil_between"][last_round]:.3f} = Theil within + Theil between")

    plt.plot(index, stats["theil"][:last_round+1], label = "avg_theil")
    plt.plot(index, stats["theil_within"][:last_round+1], label = "theil_within")
    plt.plot(index, stats["theil_between"][:last_round+1], label = "theil_between")

    plt.legend()
    plt.title("Theil across neighborhoods per round")
    plt.xlabel("Rounds")
    plt.ylabel("Theil values")
    plt.show()
    # theil within falls -> neighborhoods become more homogenous
    # theil between rises -> increased inequality between neighborhoods => segregation

    # bids and winning bids
    plt.plot(index, stats["num_bids"][:last_round+1], label = "Number of bids")
    plt.plot(index, stats["winning_bids"][:last_round+1], label = "Number of winning bids")
    plt.legend()
    plt.title("Number of bids per round")
    plt.xlabel("Rounds")
    plt.ylabel("Bids")
    plt.show()

    # calculating correlation b/w income and rents
    # FIX: rent used to be read off the first agent in the neighborhood, who could be in nonmarket housing (rent 0)
    # all houses in a neighborhood share one price, so we read it off the houses instead
    # and avg income is over residents only, so neighborhoods with no residents are dropped
    residents = resident_mask(agents)
    avg_income = np.full(n_neighborhoods, np.nan)
    rent = np.full(n_neighborhoods, np.nan)
    for i in range(n_neighborhoods):
        agent_mask = residents & (agents["neighborhood"] == i)
        house_mask = houses["neighborhood"] == i
        if np.any(agent_mask) and np.any(house_mask):
            avg_income[i] = np.mean(agents["income"][agent_mask])
            rent[i] = houses["value"][house_mask][0]
    keep = np.isfinite(avg_income) & np.isfinite(rent) & (rent > 0)
    avg_income, rent = avg_income[keep], rent[keep]

    log_income = np.log(avg_income)
    log_rent = np.log(rent)

    m_linear, b_linear = np.polyfit(avg_income, rent,1)
    correlation_linear = np.corrcoef(avg_income, rent)[0,1]
    m,b = np.polyfit(log_income, log_rent, 1) # m here -> income elasticity of demand for housing
    correlation_log = np.corrcoef(log_income, log_rent)[0,1]
    print(f"Line of best fit (linear): y = {m_linear:.4f}x + {b_linear:.4f}")
    print(f"correlation coefficient (linear): {correlation_linear:.4f}")
    print()
    print(f"Line of best fit (log): y = {m:.4f}x + {b:.4f}")
    print(f"correlation coefficient (log): {correlation_log:.4f}")

    # plotting it
    plt.scatter(avg_income, rent)
    plt.plot(avg_income, m_linear*avg_income+b_linear, label = "Line of best fit")
    plt.title("Relation between average income of a neighborhood and its rent")
    plt.xlabel("Avg income")
    plt.ylabel("Rents")
    plt.legend()
    plt.show()

    plt.scatter(log_income, log_rent)
    plt.plot(log_income, m*log_income+b, label = "Line of best fit")
    plt.title("Relation between average income of a neighborhood and its rent (in logspace)")
    plt.xlabel("Avg income")
    plt.ylabel("Rents")
    plt.legend()
    plt.show()

    # house value
    value = houses["value"]
    plt.hist(value, bins = 20, density = True)
    plt.xlabel("House value")
    plt.ylabel("Density")
    plt.title("Distribution of house value")
    plt.show()

    # plot the agents income distribution
    incomes = agents["income"]

    # Cut off at, say, the 99th percentile for visualization
    cutoff = np.percentile(incomes, 99.0)
    incomes_percentile = incomes[incomes <= cutoff]

    fig, axes = plt.subplots(2,1,figsize = (12,12)) # one plot for actual income distr, one with top 1% cut off
    axes[0].hist(incomes, bins = 500, density = True)
    axes[0].set_title("Income distribution of agents")
    axes[0].set_xlabel("Income per year in Rupees")
    axes[0].set_ylabel("Density")
    # format x-axis numbers with commas
    axes[0].xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:f}'))


    # with top 1% cut off
    axes[1].hist(incomes_percentile, bins = 500, density = True)
    axes[1].set_title("Income distribution of agents (top 1% exlcuded for a better view)")
    axes[1].set_xlabel("Income per year in Rupees")
    axes[1].set_ylabel("Density")
    axes[1].xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:f}'))

    plt.show()

"------------------------------------- same plotting function but for the mc sim ------------------------------------"
def plot_mc_stats(mc_stats, last_round, confidence=95, band="mean"): # band = "mean" (CI of the mean) or "spread" (run-to-run)
    index = np.arange(0, last_round + 1)

    def get_mean_ci(stat_name):
        return mc_mean_ci(mc_stats, stat_name, last_round, confidence, band)

    # happiness
    mean, lower, upper = get_mean_ci("happiness")
    print(f"Final happiness: {mean[-1]:.3f}%")
    plt.plot(index, mean, label="Average happiness", linewidth=2)
    plt.fill_between(index, lower, upper, alpha=0.3, label=f"{confidence}% CI")
    plt.legend()
    plt.title("Average happiness over time (Monte Carlo)")
    plt.xlabel("Rounds")
    plt.ylabel("Happiness (%)")
    plt.show()

    # nonmarket housing
    mean, lower, upper = get_mean_ci("nonmarket_housing")
    print(f"Final agents in nonmarket housing: {mean[-1]:.3f}%")
    plt.plot(index, mean, label="Nonmarket housing", linewidth=2)
    plt.fill_between(index, lower, upper, alpha=0.3, label=f"{confidence}% CI")
    plt.title("Agents in nonmarket housing over time (Monte Carlo)")
    plt.xlabel("Rounds")
    plt.ylabel("Percent of agents")
    plt.legend()
    plt.show()

    # house value
    mean, lower, upper = get_mean_ci("avg_value")
    print(f"Final average house value: {mean[-1]:.3f}")
    plt.plot(index, mean, label="Average house value", linewidth=2)
    plt.fill_between(index, lower, upper, alpha=0.3, label=f"{confidence}% CI")
    plt.legend()
    plt.title("Average house value over time (Monte Carlo)")
    plt.xlabel("Rounds")
    plt.ylabel("House value")
    plt.show()

    # churn
    mean, lower, upper = get_mean_ci("churn")
    print(f"Avg churn: {np.nanmean(mean[1:]):.3f}%")
    plt.plot(index, mean, label="Churn", linewidth=2)
    plt.fill_between(index, lower, upper, alpha=0.3, label=f"{confidence}% CI")
    plt.legend()
    plt.title("Churn per round (Monte Carlo)")
    plt.xlabel("Rounds")
    plt.ylabel("Percent of movers")
    plt.show()

    # gini
    mean, lower, upper = get_mean_ci("gini")
    print(f"Final Gini: {mean[-1]:.3f}")
    plt.plot(index, mean, label="Gini", linewidth=2)
    plt.fill_between(index, lower, upper, alpha=0.3, label=f"{confidence}% CI")
    plt.legend()
    plt.title("Avg Gini across neighborhoods per round (Monte Carlo)")
    plt.xlabel("Rounds")
    plt.ylabel("Gini index value")
    plt.show()

    # theils
    mean_theil, lower_theil, upper_theil = get_mean_ci("theil")
    mean_within, lower_within, upper_within = get_mean_ci("theil_within")
    mean_between, lower_between, upper_between = get_mean_ci("theil_between")

    print(f"Final Theil: {mean_theil[-1]:.3f}")
    print(f"Final Theil within: {mean_within[-1]:.3f}")
    print(f"Final Theil between: {mean_between[-1]:.3f}")
    print(f"Final global Theil: {mean_within[-1] + mean_between[-1]:.3f} = Theil within + Theil between")

    plt.plot(index, mean_theil, label="Avg Theil", linewidth=2)
    plt.plot(index, mean_within, label="Theil within", linewidth=2)
    plt.plot(index, mean_between, label="Theil between", linewidth=2)

    plt.fill_between(index, lower_theil, upper_theil, alpha=0.2)
    plt.fill_between(index, lower_within, upper_within, alpha=0.2)
    plt.fill_between(index, lower_between, upper_between, alpha=0.2)

    plt.legend()
    plt.title("Theil across neighborhoods per round (Monte Carlo)")
    plt.xlabel("Rounds")
    plt.ylabel("Theil values")
    plt.show()

    # bids
    mean_bids, lower_bids, upper_bids = get_mean_ci("num_bids")
    mean_winners, lower_winners, upper_winners = get_mean_ci("winning_bids")

    plt.figure(figsize=(10, 6))
    plt.plot(index, mean_bids, label="Number of bids", linewidth=2)
    plt.plot(index, mean_winners, label="Number of winning bids", linewidth=2)

    plt.fill_between(index, lower_bids, upper_bids, alpha=0.2)
    plt.fill_between(index, lower_winners, upper_winners, alpha=0.2)

    plt.legend()
    plt.title("Number of bids per round (Monte Carlo)")
    plt.xlabel("Rounds")
    plt.ylabel("Bids")
    plt.show()

"----------------------------------------- policy vs baseline, round by round ---------------------------------------"
# used by evaluate_policy. metrics = list of (stat name, axis label)
def plot_policy_comparison(mc_b, mc_p, last_round, metrics, confidence = 95, band = "mean"):
    index = np.arange(last_round + 1)
    for stat, label in metrics:
        mean_b, low_b, up_b = mc_mean_ci(mc_b, stat, last_round, confidence, band)
        mean_p, low_p, up_p = mc_mean_ci(mc_p, stat, last_round, confidence, band)

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

"------------------------------------ policy vs baseline, happiness by income bracket ------------------------------"
# happiness by income bracket at the end of each run, baseline vs policy
# this is the 'who does the policy reach' plot. bars = mean across runs, error bars = CI across runs
def plot_policy_bracket_happiness(mc_b, mc_p, income_cutoff = 2, confidence = 95, band = "mean"):
    alpha = (100 - confidence) / 100
    brackets = np.arange(N_BRACKETS)
    width = 0.4
    plt.figure(figsize=(10, 6))
    for offset, mc, label in [(-width/2, mc_b, "Baseline"), (width/2, mc_p, "Affordable housing policy")]:
        data = mc["bracket_happiness"]
        mean = np.nanmean(data, axis=0)
        n = np.sum(np.isfinite(data), axis=0)
        if band == "spread":
            lo = np.nanpercentile(data, alpha/2*100, axis=0)
            hi = np.nanpercentile(data, (1-alpha/2)*100, axis=0)
        else: # CI of the mean
            half = student_t.ppf(1 - alpha/2, np.maximum(n-1, 1)) * np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)
            lo, hi = mean - half, mean + half
        plt.bar(brackets + offset, mean, width, yerr=[mean-lo, hi-mean], capsize=3, label=label)
    plt.axvline(income_cutoff + 0.5, color="grey", linestyle="--", linewidth=1) # eligible brackets are to the left
    plt.title("Happiness by income bracket: Baseline vs Policy")
    plt.xlabel("Income bracket")
    plt.ylabel("Happiness (%)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.show()

"--------------------------------------------- welfare: who gains, who loses ----------------------------------------"
# used by report_welfare in policy.py
def plot_welfare(welfare, income_cutoff = 2, confidence = 95):
    brackets = np.arange(N_BRACKETS)
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    mean, lo, hi = runs_mean_ci(welfare["ev_median"], confidence) # median agent in each bracket, CI across runs
    axes[0].bar(brackets, mean, yerr = [mean - lo, hi - mean], capsize = 3)
    axes[0].axhline(0, color = "black", linewidth = 0.8)
    axes[0].axvline(income_cutoff + 0.5, color = "grey", linestyle = "--", linewidth = 1) # eligible brackets to the left
    axes[0].set_title("Welfare effect of the policy by income bracket")
    axes[0].set_xlabel("Income bracket")
    axes[0].set_ylabel("Median equivalent variation (% of income)")
    axes[0].grid(alpha = 0.3, axis = "y")

    gain = np.nanmean(welfare["gain_pct"], axis = 0)
    lose = np.nanmean(welfare["lose_pct"], axis = 0)
    axes[1].bar(brackets, gain, label = "Better off")
    axes[1].bar(brackets, -lose, label = "Worse off")
    axes[1].axhline(0, color = "black", linewidth = 0.8)
    axes[1].axvline(income_cutoff + 0.5, color = "grey", linestyle = "--", linewidth = 1)
    axes[1].set_title("Who gains and who loses")
    axes[1].set_xlabel("Income bracket")
    axes[1].set_ylabel("% of bracket (losers shown below zero)")
    axes[1].legend()
    axes[1].grid(alpha = 0.3, axis = "y")

    plt.tight_layout()
    plt.show()
