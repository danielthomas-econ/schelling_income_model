## agent behavior
divide agents into different income brackets in a lognormal distribution. we have deciles until the highest tiers, where it becomes 90-95%ile, 95-99%ile, 99%ile+ (done ✅)

perhaps add a pareto tail to the top earners to make it more realistic, because the sim vastly underestimates the number of high earners (not done yet, might do in the future)

give each agent a location in a 10x10 grid. each 1x1 square is called a neighborhood, assign each one a number (done ✅)

agents want at least x% of their neighbors (other agents living in the neighborhood) to be of an equal or higher income category than them (done ✅)

we go through every agent and check their happiness status. one massive list of arrays with True/False to check if agent is happy (done ✅)

everyone on the list tries to buy a house in a neighborhood they would be happy in.

repeat until everyone is happy

potential idea?: incorporate some slight popln growth rate overall for the city

## rents
rents start at a uniform level.

the homes are bid on by the agents.

#### bids (done ✅)
bidding logic involves social WTP, so its a variable we can vary and set counterfactuals for

each agent computes max bid as $B_{i} = \min\left( (\beta + \lambda U_{i})Y_{i},\delta Y_{i} \right)$, 
where:
$Y_{i} =$ agent $i$'s income
$\beta =$ baseline budget fraction (like 0.3)
$U_{i} =$ social utility from living in the neighborhood being bid on (0-1 scale)
$\lambda =$ marginal WTP for 1 unit of social utility 
$\delta =$ absolute affordability cap (maybe 0.6?)

term one: some prpn of $Y_{i}$ + WTP for social utility
term two: max percent of income agent is willing to spend on rent
take min so that we don't have unrealistically high bids (like entire $Y_{i}$)

bid zero if the agent is happy

#### utility function: (done ✅)
cobb douglas utility, then normalize it to be 0-1. The utility of neighborhood $k$ for agent $i$ is calculated as
$U_{i,k} = q_{i,k}^{\theta} \cdot c_{i,k} ^{1-\theta}$,
where:
$q_{i,k} =$ neighborhood quality for agent i in neighborhood k, which is given by proportion of agents in $k$ with >= income bracket
$c_{i,k} =$ proportion of income left for consumption on goods other than housing $(Y_{i} - P_{i})/Y_i$.
$\theta=$ how much agent values neighborhood quality relative to consumption. maybe $\theta_{i} \sim \text{Uniform}(0.6,0.8)$ for heterogenous agents
$\theta$ close to 1 is more like Schelling behavior (neighborhood quality only matters), close to 0 is like a textbook economic agent.

\> currently, we calculate $c_{i,k}$ based on current rent paid in neighborhood $j$, not rent in the prospective neighborhood $k$. I need to figure out how to do this.

#### price update rule: (done ✅)
we update the house prices for all units in the neighborhood using 
$P_{j,t+1} = P_{j,t} + \alpha \left(B - P_{j,t} \right) + \epsilon_{t+1}$,
where:
$B_{j} =$ avg winning bid this round
$P_{j} = $  the current rent of house $j$
$\alpha = $  multiplicative constant giving speed of convergence to average in some sense
$\epsilon =$ noise term $\sim N(0,\sigma^{2})$, where $\sigma = 0.02\cdot \bar{P}$ (average price)

impose a price floor: house price >= 0

#### agents being priced out: (done ✅)
when rent increases, its possible that staying in the current neighborhood is no longer feasible for an agent
We define the agent's max WTP for rent in neighborhood $j$ as
$B_{i}^{\text{stay}}= \text{min}((\beta + \lambda U_{i,j})Y_{i}, \delta Y_i)$


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
\> add a `common.py` file with worldwide constants and agents/houses datatypes to have everything changed from one central location
\> analyze through the lens of a willingness-to-pay (WTP) premium for social utility (moving to a preferred neighborhood). we can change $\theta$ and $\lambda$ to range from 'economic' agents (pure rational) for low values to 'sociological' agents (derive greater utilty from neighborhood)
\> scatter plot showing correlation between income bracket and avg income of current neighborhood
\> look at government intervention and conduct rcts to test the impact

### concerning things
looking at the distribution of income brackets, it looks roughly uniformly distributed until the higher income levels.
i think directly doing probs = np.array(percentiles)/100 and using probs as a cdf could be the cause

\>  Correlation with Bidding Success: Analyze the correlation between an agent's c_term and their success rate in winning bids for socially desirable neighborhoods (i.e., neighborhoods where q_diff is high). You should find that agents with lower c_term values (due to high current rent burdens) are less likely to win bids, even for neighborhoods that would make them happy.


The Constraint: Because most agents desire higher-income neighbors, and there are inherently few high-income individuals, it becomes a competitive, zero-sum game for these desirable neighbors. Not everyone can achieve their high x% threshold for "better" neighbors, leading to the observed 1-x happiness convergence. The "proportion of the population that can realistically achieve this preference" is limited by the availability of the desired income brackets.

### things to test in the model
\> test the model without income brackets and see if that gives us more realistic results
\> test with +-1 income bracket instead of >= income bracket too
