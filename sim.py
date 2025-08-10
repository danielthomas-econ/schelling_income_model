from src.agents import generate_agents, neighborhood_lists, find_income_brackets
import numpy as np

incomes, brackets, locations, neighborhood = generate_agents()
num_agents = len(incomes)
cutoffs = find_income_brackets()

# agent wants {happiness_percent}% of people in his neighborhood to be of the same income bracket or higher
# PROBLEM!!!!! the structure of happiness_list is not the same as neighborhood list, fix that
def is_happy(cutoffs = cutoffs, neighborhood = neighborhood, happiness_percent = 0.5):
    neighborhood_list = neighborhood_lists(neighborhood=neighborhood)
    happiness_list = np.empty(len(neighborhood))
    for n in neighborhood_list:
        # convert the list from agent numbers to agent income brackets
        n_income = incomes[n]
        brackets = np.searchsorted(cutoffs, n_income) - 1
        total = len(n)
        for agent in range(total):
            good = np.sum(brackets >= brackets[agent])
            if (good/total) >= happiness_percent:
                happiness_list[n[agent]] = 1
            else:
                happiness_list[n[agent]] = 0
    return happiness_list

print(is_happy())