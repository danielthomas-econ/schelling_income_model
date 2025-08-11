## agent behavior
divide agents into different income brackets in a lognormal distribution. we have deciles until the highest tiers, where it becomes 90-95%ile, 95-99%ile, 99%ile+ (done ✅)

perhaps add a pareto tail to the top earners to make it more realistic, because the sim vastly underestimates the number of high earners

give each agent a location in a 10x10 grid. each 1x1 square is called a neighborhood, assign each one a number (done ✅)

agents want at least x% of their neighbors (other agents living in the neighborhood) to be of an equal or higher income category than them (done ✅)

we go through every agent and check their happiness status. one massive list of arrays with True/False to check if agent is happy (done ✅)

everyone on the list tries to buy a house in a neighborhood they would be happy in.

repeat until everyone is happy

potential idea?: incorporate some slight popln growth rate overall for the city

## rents
rents start at a uniform level.

the homes are bid on by the agents.

#### bids
bidding logic involves social WTP, so its a variable we can vary and set counterfactuals for

each agent computes max bid as $B_{i} = \min\left( \beta Y_{i} + \lambda U_{i},\delta Y_{i} \right)$, 
where:
$Y_{i} =$ agent $i$'s income
$\beta =$ baseline budget fraction (like 0.3)
$U_{i} =$ social utility from living in the neighborhood being bid on (0-1 scale)
$\lambda =$ marginal WTP for 1 unit of social utility 
$\delta =$ absolute affordability cap (maybe 0.6?)

term one: some prpn of $Y_{i}$ + WTP for social utility
term two: max percent of income agent is willing to spend on rent
take min so that we don't have unrealistically high bids (like entire $Y_{i}$)

#### utility function:
cobb douglas utility, then normalize it to be 0-1 
$U_{i,j} = q_{i,j}^{\theta} \cdot c_{i,j} ^{1-\theta}$,
where:
$q_{i,j} =$ neighborhood quality for agent i in neighborhood j
$q_{j} =$ % of residents with >= income bracket than agent i
$c_{i,j} =$ consumption on goods other than housing $(Y_{i} - P{i})$.
$\theta=$ how much agent values neighborhood quality relative to consumption. maybe $\theta_{i} \sim \text{Uniform}(0.6,0.8)$ for heterogenous agents
$\theta$ close to 1 is more like Schelling behavior (neighborhood quality only matters), close to 0 is like a textbook economic agent.

#### price update rule:
we update the house prices for all units in the neighborhood using 
$P_{j,t+1} = P_{j,t} + \alpha \left(B_{j} - P_{j,t} \right)$,
where:
$B_{j} =$ avg winning bid in neighborhood j
$P_{j}$ be the current rent of house $j$
$\alpha$ be some randomly distributed multiplicative constant

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
analyze through the lens of a willingness-to-pay (WTP) premium for social utility (moving to a preferred neighborhood).
look at government intervention and conduct rcts to test the impact

### concerning things
looking at the distribution of income brackets, it looks roughly uniformly distributed until the higher income levels.
i think directly doing probs = np.array(percentiles)/100 and using probs as a cdf could be the cause

neighborhoods array seems to be uniformly distributed (intended), but it allocates too many people in neighborhoods 0-5 and too few in 95-100