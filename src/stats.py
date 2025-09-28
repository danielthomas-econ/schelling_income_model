"Functions to calculate all sorts of stats for our sim"

import numpy as np

def initialize_stats(num_rounds, num_neighborhoods):
    # prev_house used to be stored here, but it was of the shape (num_rounds, num_agents) and was a HUGE memory hog
    # now we dont have a field for prev_house here, but its calculated on the fly for each round at the end of get_stats
    stats_dtype = np.dtype([
        ("avg_value", np.float32),
        ("avg_income", [("income", np.float32, (num_neighborhoods,)), # we need to add an extra shape argument, otherwise this will only retain the value for the last neighborhood because of the shape mismatch
                        ("neighborhood", np.int8, (num_neighborhoods,))]), # avg income never changes, so we want to look at avg income by neighborhood
        ("homelessness", np.float32), # percentage
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

def get_stats(stats, agents, houses, current_round, prev_house = None):
    n_agents = agents.size
    n_neighborhoods = np.max(agents["neighborhood"])+1
    global_avg_income = np.mean(agents["income"])

    stats["avg_value"][current_round] = np.mean(houses["value"])
    stats["homelessness"][current_round] = np.sum(agents["neighborhood"]==-1)*100/n_agents
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

# the next two functions are the same as the two above, but with an extra dimension for number of runs
# only diff is we can store prev_house in stats itself here
def initialize_mc_stats(num_runs, num_rounds, num_agents, num_neighborhoods):

    stats_dtype = np.dtype([
        ("avg_value", np.float32, (num_runs, num_rounds)),
        ("avg_income", [("income", np.float32, (num_runs, num_rounds, num_neighborhoods)),
                        ("neighborhood", np.int8, (num_runs, num_rounds, num_neighborhoods))]),
        ("homelessness", np.float32, (num_runs, num_rounds)),
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


def get_mc_stats(mc_stats, agents, houses, run_id, current_round):
    n_agents = agents.size
    n_neighborhoods = np.max(agents["neighborhood"]) + 1
    global_avg_income = np.mean(agents["income"])

    mc_stats["avg_value"][run_id, current_round] = np.mean(houses["value"])
    mc_stats["homelessness"][run_id, current_round] = np.sum(agents["neighborhood"] == -1) * 100 / n_agents
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