from src.agents import generate_agents, neighborhood_lists, find_income_brackets
import numpy as np

incomes, brackets, locations, neighborhood = generate_agents()
num_agents = len(incomes)
cutoffs = find_income_brackets()
