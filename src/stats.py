import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # to remove 1e6 base from the x axis on plots

"-------------------------------- initialize a structured array to hold all our stats -------------------------------"
def initialize_stats(num_rounds, num_neighborhoods):
    # prev_house used to be stored here, but it was of the shape (num_rounds, num_agents) and was a HUGE memory hog
    # now we dont have a field for prev_house here, but its calculated on the fly for each round at the end of get_stats
    stats_dtype = np.dtype([
        ("avg_value", np.float32),
        ("avg_income", [("income", np.float32, (num_neighborhoods,)), # we need to add an extra shape argument, otherwise this will only retain the value for the last neighborhood because of the shape mismatch
                        ("neighborhood", np.int8, (num_neighborhoods,))]), # avg income never changes, so we want to look at avg income by neighborhood
        ("nonmarket_housing", np.float32), # percentage
        ("vacancies", np.float32), # percentage
        ("happiness", np.float32), # percentage
        ("gini", np.float32), # percentage
        ("theil", np.float32), # >= 0, no upper bound
        ("theil_within", np.float32),
        ("theil_between", np.float32),
        ("dissimilarity", np.float32),
        ("churn", np.float32),
        ("num_bids", np.uint32),
        ("winning_bids", np.uint32)
    ])
    stats = np.zeros(num_rounds, dtype=stats_dtype)
    stats.fill(0)
    return stats

"---------------------------------------------- updates the stats array ---------------------------------------------"
def get_stats(stats, agents, houses, current_round, prev_house = None):
    n_agents = agents.size
    n_neighborhoods = np.max(agents["neighborhood"])+1
    global_avg_income = np.mean(agents["income"])

    stats["avg_value"][current_round] = np.mean(houses["value"])
    stats["nonmarket_housing"][current_round] = np.sum(agents["nonmarket_housing"]==True)*100/n_agents
    stats["happiness"][current_round] = np.sum(agents["happy"])*100/n_agents
    stats["vacancies"][current_round] = np.sum(houses["tenant"]==-1)*100/n_agents

    neighborhood_ginis = []
    neighborhood_theils = []
    theil_within = 0
    theil_between = 0
    # stats for each neighborhood per round
    for i in range(n_neighborhoods):
        # number of agents in the neighborhood might be needed a lot, so its here in the start
        mask = agents["neighborhood"] == i
        agents_in_i = np.sum(mask)

        # avg income
        stats["avg_income"]["income"][current_round, i] = np.mean(agents["income"][mask])
        stats["avg_income"]["neighborhood"][current_round, i] = i

        # gini
        if agents_in_i > 0:
            sorted_income = np.sort(agents["income"][mask])
            gini_val = (2 * np.sum((np.arange(1,agents_in_i+1)*sorted_income))) / (agents_in_i * np.sum(sorted_income)) - (agents_in_i+1)/agents_in_i
            neighborhood_ginis.append(gini_val)
        else:
            neighborhood_ginis.append(np.nan) # set undefined gini if no agents in the neighborhood to prevent divide by zero issues in edge cases

        # theil
        if agents_in_i > 0:
            nb_income = agents["income"][mask]
            nb_avg_income = np.mean(nb_income)
            theil_val = (np.sum(nb_income/nb_avg_income * np.log(nb_income/nb_avg_income))) / agents_in_i
            neighborhood_theils.append(theil_val)

            theil_within += theil_val * (agents_in_i)/n_agents
            theil_between += (agents_in_i/n_agents) * (nb_avg_income/global_avg_income) * np.log(nb_avg_income/global_avg_income)
        else:
            neighborhood_theils.append(np.nan) # again, undefined theil if nobody lives there

    stats["gini"][current_round] = np.nanmean(neighborhood_ginis) * 100 # nanmean ignores nan values when calculating mean
    stats["theil"][current_round] = np.nanmean(neighborhood_theils)
    stats["theil_within"][current_round] = theil_within
    stats["theil_between"][current_round] = theil_between

    # churn calculations
    if current_round > 0 and prev_house is not None:
        # look at differences b/w current houses and last round's house allocations
        moved = agents["house"] != prev_house
        stats["churn"][current_round] = np.sum(moved) * 100/n_agents # churn = % of movers
    else:
        stats["churn"][current_round] = 0.0

    return stats, agents["house"].copy() # -> prev_house for the next round

"----------------------------------------- initializing for monte carlo runs ----------------------------------------"
# the next two functions are the same as the two above, but with an extra dimension for number of runs
# only diff is we can store prev_house in stats itself here
def initialize_mc_stats(num_runs, num_rounds, num_agents, num_neighborhoods):

    stats_dtype = np.dtype([
        ("avg_value", np.float32, (num_runs, num_rounds)),
        ("avg_income", [("income", np.float32, (num_runs, num_rounds, num_neighborhoods)),
                        ("neighborhood", np.int8, (num_runs, num_rounds, num_neighborhoods))]),
        ("nonmarket_housing", np.float32, (num_runs, num_rounds)),
        ("vacancies", np.float32, (num_runs, num_rounds)),
        ("happiness", np.float32, (num_runs, num_rounds)),
        ("gini", np.float32, (num_runs, num_rounds)),
        ("theil", np.float32, (num_runs, num_rounds)),
        ("theil_within", np.float32, (num_runs, num_rounds)),
        ("theil_between", np.float32, (num_runs, num_rounds)),
        ("dissimilarity", np.float32, (num_runs, num_rounds)),
        ("prev_house", np.int32, (num_runs, num_agents)),
        ("churn", np.float32, (num_runs, num_rounds)),
        ("num_bids", np.uint32, (num_runs, num_rounds)),
        ("winning_bids", np.uint32, (num_runs, num_rounds))
    ])
    stats = np.zeros((), dtype=stats_dtype)  # single structured record
    return stats

"------------------------------------------- the actual mc stats collector ------------------------------------------"
def get_mc_stats(mc_stats, agents, houses, run_id, current_round):
    n_agents = agents.size
    n_neighborhoods = np.max(agents["neighborhood"]) + 1
    global_avg_income = np.mean(agents["income"])

    mc_stats["avg_value"][run_id, current_round] = np.mean(houses["value"])
    mc_stats["nonmarket_housing"][run_id, current_round] = np.sum(agents["nonmarket_housing"] == True) * 100 / n_agents
    mc_stats["happiness"][run_id, current_round] = np.sum(agents["happy"]) * 100 / n_agents
    mc_stats["vacancies"][run_id, current_round] = np.sum(houses["tenant"] == -1) * 100 / n_agents

    neighborhood_ginis = []
    neighborhood_theils = []
    theil_within = 0
    theil_between = 0

    for i in range(n_neighborhoods):
        mask = agents["neighborhood"] == i
        agents_in_i = np.sum(mask)

        mc_stats["avg_income"]["income"][run_id, current_round, i] = np.mean(agents["income"][mask])
        mc_stats["avg_income"]["neighborhood"][run_id, current_round, i] = i

        if agents_in_i > 0:
            # gini
            sorted_income = np.sort(agents["income"][mask])
            gini_val = (2 * np.sum((np.arange(1, agents_in_i + 1) * sorted_income))) / \
                       (agents_in_i * np.sum(sorted_income)) - (agents_in_i + 1) / agents_in_i
            neighborhood_ginis.append(gini_val)

            # theil
            nb_income = agents["income"][mask]
            nb_avg_income = np.mean(nb_income)
            theil_val = np.sum(nb_income / nb_avg_income * np.log(nb_income / nb_avg_income)) / agents_in_i
            neighborhood_theils.append(theil_val)

            theil_within += theil_val * (agents_in_i) / n_agents
            theil_between += (agents_in_i / n_agents) * (nb_avg_income / global_avg_income) * \
                             np.log(nb_avg_income / global_avg_income)
        else:
            neighborhood_ginis.append(np.nan)
            neighborhood_theils.append(np.nan)

    mc_stats["gini"][run_id, current_round] = np.nanmean(neighborhood_ginis) * 100
    mc_stats["theil"][run_id, current_round] = np.nanmean(neighborhood_theils)
    mc_stats["theil_within"][run_id, current_round] = theil_within
    mc_stats["theil_between"][run_id, current_round] = theil_between

    # churn
    if current_round > 0:
        moved = agents["house"] != mc_stats["prev_house"][run_id]
        mc_stats["churn"][run_id, current_round] = np.sum(moved) * 100 / n_agents
    else:
        mc_stats["churn"][run_id, current_round] = 0.0

    mc_stats["prev_house"][run_id] = agents["house"]

    return mc_stats

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
    print(f"Avg churn: {np.mean(stats["churn"]):.3f}%")
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
    avg_income = np.zeros(n_neighborhoods)
    rent = np.zeros(n_neighborhoods) # all houses in a neighborhood share the same price, so avg is meaningless

    for i in range(n_neighborhoods):
        agent_mask = agents["neighborhood"] == i
        if len(agent_mask) > 0:
            avg_income[i] = np.mean(agents["income"][agent_mask])
            if rent[i] == 0:
                valid = agents["rent_paid"][agent_mask]
                rent[i] = valid[0]
        else:
            avg_income[i] = 0
        
    log_income = np.log(avg_income)
    log_rent = np.zeros_like(log_income)
    for i in range(len(rent)):
        if rent[i] > 0:
            log_rent[i] = np.log(rent[i])
        else:
            log_rent[i] = 0

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
def plot_mc_stats(mc_stats, last_round, confidence=95): # confidence lets us plot CI intervals too
    index = np.arange(0, last_round + 1)
    
    # CI calculations
    alpha = (100 - confidence) / 100
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # write this once so we dont have to redo it all the time
    def get_mean_ci(stat_name):
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
    print(f"Avg churn: {np.mean(mean):.3f}%")
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
   