## agent behavior
divide agents into different income brackets in a lognormal distribution. we have deciles until the highest tiers, where it becomes 90-95%ile, 95-99%ile, 99%ile+ (done ✅)

perhaps add a pareto tail to the top earners to make it more realistic, because the sim vastly underestimates the number of high earners

give each agent a location in a 10x10 grid. each 1x1 square is called a neighborhood, assign each one a number (done ✅)

agents want at least x% of their neighbors (other agents living in the neighborhood) to be of an equal or higher income category than them

we go through every agent and check their happiness status. if they aren't happy, we add them to a list.

everyone on the list tries to buy a house in a neighborhood they would be happy in.

repeat until everyone is happy

potential idea?: incorporate some slight popln growth rate overall for the city

## house prices
house prices start at a uniform level.

the houses are bid on by the agents. the agent with the highest income gets the house. 

final bid is the agent's budget, which is a certain multiple of their income.

price update rule:
let $D_{j}$ be number of bids for house $j$.
let $S_{j}$ be number of vacanies in the neighborhood of house $j$.
let $P_{j}$ be the current price of house $j$
let $\alpha$ be some randomly distributed multiplicative constant
then, we update the house prices for all units in the neighborhood using $P_{j,t+1} = P_{j, t} + \alpha \cdot (D_{j}-S_{j})$

even if a unit didnt receive a bid, its price goes up if the overall demand for that area was high (spillage) and vv.

impose a price floor: house price >= 0

## main algorithm
each neighborhood = one grid cell
each cell has a fixed number of houses
each agent occupies one unit of housing
agent moves if they are not happy
agent can only afford certain neighborhoods
agent places one bid a round. stays in their current unit if their bid fails

happiness check -> bids -> price update -> assignment -> happiness check

stop if all agents are happy

### future plans
look at government intervention and conduct rcts to test the impact

### concerning things
looking at the distribution of income brackets, it looks roughly uniformly distributed until the higher income levels.
i think directly doing probs = np.array(percentiles)/100 and using probs as a cdf could be the cause

neighborhoods array seems to be uniformly distributed (intended), but it allocates too many people in neighborhoods 0-5 and too few in 95-100