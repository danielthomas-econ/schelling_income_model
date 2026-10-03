import numpy as np
from scipy.stats import t as student_t
from .common import resident_mask, N_BRACKETS

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
        ("churn", np.float32),
        ("num_bids", np.uint32),
        ("winning_bids", np.uint32)
    ])
    stats = np.zeros(num_rounds, dtype=stats_dtype)
    stats.fill(0)
    return stats

"------------------------------- inequality stats shared by get_stats and get_mc_stats ------------------------------"
# both stats collectors used to have their own copy of this loop. it now lives in one place
# only residents (see COUNT_NONMARKET in common.py) count towards a neighborhood, so agents in nonmarket housing
# don't get counted in a neighborhood they've been evicted from
def neighborhood_inequality(agents, n_neighborhoods):
    residents = resident_mask(agents)
    n_res = np.sum(residents)
    avg_income = np.full(n_neighborhoods, np.nan)
    ginis = np.full(n_neighborhoods, np.nan) # undefined gini/theil if nobody lives there
    theils = np.full(n_neighborhoods, np.nan)
    theil_within = 0.0
    theil_between = 0.0
    if n_res == 0: # everyone is in nonmarket housing (only possible before any allocation)
        return avg_income, np.nan, np.nan, np.nan, np.nan

    global_avg_income = np.mean(agents["income"][residents])
    for i in range(n_neighborhoods):
        mask = residents & (agents["neighborhood"] == i)
        agents_in_i = np.sum(mask)
        if agents_in_i == 0:
            continue
        nb_income = agents["income"][mask]
        nb_avg_income = np.mean(nb_income)
        avg_income[i] = nb_avg_income

        # gini
        sorted_income = np.sort(nb_income)
        ginis[i] = (2 * np.sum((np.arange(1,agents_in_i+1)*sorted_income))) / (agents_in_i * np.sum(sorted_income)) - (agents_in_i+1)/agents_in_i

        # theil, weights are shares of the resident population so that within + between = global theil
        theils[i] = (np.sum(nb_income/nb_avg_income * np.log(nb_income/nb_avg_income))) / agents_in_i
        theil_within += theils[i] * agents_in_i/n_res
        theil_between += (agents_in_i/n_res) * (nb_avg_income/global_avg_income) * np.log(nb_avg_income/global_avg_income)

    return avg_income, np.nanmean(ginis) * 100, np.nanmean(theils), theil_within, theil_between

"---------------------------------------------- updates the stats array ---------------------------------------------"
def get_stats(stats, agents, houses, current_round, prev_house = None):
    n_agents = agents.size
    n_neighborhoods = stats["avg_income"]["income"].shape[1]

    stats["avg_value"][current_round] = np.mean(houses["value"])
    stats["nonmarket_housing"][current_round] = np.sum(agents["nonmarket_housing"]==True)*100/n_agents
    stats["happiness"][current_round] = np.sum(agents["happy"])*100/n_agents
    stats["vacancies"][current_round] = np.sum(houses["tenant"]==-1)*100/n_agents

    avg_income, gini, theil, theil_within, theil_between = neighborhood_inequality(agents, n_neighborhoods)
    stats["avg_income"]["income"][current_round] = avg_income
    stats["avg_income"]["neighborhood"][current_round] = np.arange(n_neighborhoods)
    stats["gini"][current_round] = gini
    stats["theil"][current_round] = theil
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
        ("prev_house", np.int32, (num_runs, num_agents)),
        ("churn", np.float32, (num_runs, num_rounds)),
        ("num_bids", np.uint32, (num_runs, num_rounds)),
        ("winning_bids", np.uint32, (num_runs, num_rounds)),
        # the last round each run reached. runs can end at different rounds when converge = True,
        # and we use this (instead of 'value == 0') to know which runs are still alive at a given round
        ("last_round", np.int32, (num_runs,)),
        # % of agents happy in each income bracket at the end of each run
        ("bracket_happiness", np.float32, (num_runs, N_BRACKETS)),
    ])
    stats = np.zeros((), dtype=stats_dtype)  # single structured record
    stats["last_round"] = -1 # -1 => run hasn't finished yet
    stats["bracket_happiness"] = np.nan
    return stats

"------------------------------------------- the actual mc stats collector ------------------------------------------"
def get_mc_stats(mc_stats, agents, houses, run_id, current_round):
    n_agents = agents.size
    n_neighborhoods = mc_stats["avg_income"]["income"].shape[2]

    mc_stats["avg_value"][run_id, current_round] = np.mean(houses["value"])
    mc_stats["nonmarket_housing"][run_id, current_round] = np.sum(agents["nonmarket_housing"] == True) * 100 / n_agents
    mc_stats["happiness"][run_id, current_round] = np.sum(agents["happy"]) * 100 / n_agents
    mc_stats["vacancies"][run_id, current_round] = np.sum(houses["tenant"] == -1) * 100 / n_agents

    avg_income, gini, theil, theil_within, theil_between = neighborhood_inequality(agents, n_neighborhoods)
    mc_stats["avg_income"]["income"][run_id, current_round] = avg_income
    mc_stats["avg_income"]["neighborhood"][run_id, current_round] = np.arange(n_neighborhoods)
    mc_stats["gini"][run_id, current_round] = gini
    mc_stats["theil"][run_id, current_round] = theil
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

"-------------------------------------------- mean and CI across MC runs --------------------------------------------"
# FIX: the old version threw away every value equal to 0, assuming 0 meant 'this run has already ended'
# but churn, nonmarket housing, num_bids etc can genuinely be 0, so those rounds were silently dropped
# now we use mc_stats["last_round"] to know which runs are still alive at each round
def mc_mean_ci(mc_stats, stat_name, last_round, confidence = 95, band = "mean"):
    # band = "mean":   confidence interval for the MEAN across runs, mean ± t * sd / sqrt(n). this is what a
    #                  'confidence interval' means, and it's what the shaded area should show when you compare two means
    # band = "spread": the middle `confidence`% of individual runs (the old behavior). this is not a CI, it's the
    #                  run-to-run spread, and it does NOT shrink as you add runs, which is why the old bands were so wide
    alpha = (100 - confidence) / 100
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    data = mc_stats[stat_name][:, :last_round + 1].astype(np.float64)
    run_ends = mc_stats["last_round"]

    mean = np.full(last_round + 1, np.nan)
    lower = np.full(last_round + 1, np.nan)
    upper = np.full(last_round + 1, np.nan)
    for t in range(last_round + 1):
        alive = run_ends >= t # runs that reached round t
        vals = data[alive, t]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean[t] = np.mean(vals)
        if band == "spread":
            lower[t] = np.percentile(vals, lower_percentile)
            upper[t] = np.percentile(vals, upper_percentile)
        elif vals.size > 1:
            half = student_t.ppf(1 - alpha / 2, vals.size - 1) * np.std(vals, ddof = 1) / np.sqrt(vals.size)
            lower[t] = mean[t] - half
            upper[t] = mean[t] + half
        else:
            lower[t] = upper[t] = mean[t]
    return mean, lower, upper

"--------------------------------- mean and CI across runs for a (runs, ...) array ---------------------------------"
# same idea as mc_mean_ci, for per-run results that aren't indexed by round (eg the welfare numbers in policy.py)
def runs_mean_ci(data, confidence = 95):
    # mean across runs (axis 0) with a t confidence interval
    data = np.asarray(data, dtype = np.float64)
    n = np.sum(np.isfinite(data), axis = 0)
    mean = np.nanmean(data, axis = 0)
    alpha = (100 - confidence) / 100
    with np.errstate(invalid = "ignore", divide = "ignore"):
        half = student_t.ppf(1 - alpha/2, np.maximum(n - 1, 1)) * np.nanstd(data, axis = 0, ddof = 1) / np.sqrt(n)
    return mean, mean - half, mean + half
