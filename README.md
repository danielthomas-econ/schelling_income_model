# A Schelling inspired income segregation model

***Summary:*** This project is an agent based model that looks at emergent segregation patterns. Each agent has a Schelling preference - they want $x\%$ or more of their neighbors to be of an equal or higher income bracket than them. If this is satisfied, they are classified as 'happy'. *Agents will use their income to bid on neighborhoods that make them happy, which results in rising segregation as agents cluster up based on their incomes.*

In short, it is a modification of Schelling's original model, but preferences are based on income brackets and with a housing market to add movement constraints for each agent.

## Table of contents:
- [Why have I built this?](#why-have-i-built-this)
- [How to run the sim](#how-to-run-the-sim)
- [Project structure](#project-structure)
- [Agent logic](#agent-logic)
  - [Bidding](#bidding)
  - [Utility function](#utility-function)
- [Housing logic](#housing-logic)
  - [Price update rule](#price-update-rule)
  - [Pricing out agents](#pricing-out-agents)
	- [What happens to agents without a home?](#what-happens-to-agents-without-a-home)
- [Main algorithm](#main-algorithm)
- [Optimizations used](#optimizations-used)
  - [Benchmarks](#benchmarks)
- [Stats](#stats)
- [Parameters](#parameters)
- [License](#license)

## Why have I built this?
Using this model, I want to look at a few research questions:
1) Using *Shapley decomposition* to attribute the impact of market factors (house prices) and social factors (agent preferences) on the total segregation effect.
2) Using machine learning (like `scikit-learn`) to optimize model parameters to maximize happiness and minimize segregation.
3) Looking at the impact of various model parameters on aggregate happiness and segregation.
4) Observing the impacts of policies through conducting RCTs.

## How to run the sim
Start by running `pip install -r requirements.txt` to get all the dependencies installed. Then, simply open `plots.ipynb` and run the sim.

## Project structure
~~~
plots.ipynb # Visualizing stats for the sim
src/
├── common.py # Constants and configurations
├── agents.py # Agent generation and behavior
├── bidding.py # Bidding logic and utility functions
├── houses.py # Housing market operations
├── stats.py # Statistics and metrics
└── sim.py # Main simulation loop
data/
├── income_quantile_delhi.csv # Income distribution data
~~~
## Agent logic
Each agent is given an income, drawn from `/data/income_quantile_delhi.csv`. This is a quantile function I've approximated for Delhi's income distribution using the CMIE consumer pyramid dataset. *Based on their incomes, agents are divided into income brackets,* given by `PERCENTILES` in `common.py`. Our agents have a social preference: **they want $x\%$ of agents in their neighborhood to have $\geq$ income bracket compared to them.** If this is true, the agent is classified as happy. 

### Bidding
If an agent is unhappy, he will bid to move into another neighborhood. Agent $i$ computes his bid for neighborhood $k$ using the formula
$$
B_{i,k}= \text{min}((\beta+\gamma U_{i,k})Y_{i}, \delta Y_{i})
$$
where:
- $Y_{i}=$ agent $i$'s income
- $\beta =$ baseline budget fraction he's willing to bid
- $U_{i,k}=$ utility agent $i$ derives from living in neighborhood $k$ (normalized to be between $0$ and $1$)
- $\gamma=$ marginal WTP for 1 unit of utility
- $\delta=$ maximum percent of his income he is willing to bid for rent

The first term tells us how much the agent is willing to pay based on the utility derived from the neighborhood $k$. Sometimes, this ends up being unrealistically high, so $\delta Y_{i}$ caps his total bid to a reasonable amount ($60\%$ of income by default).

### Utility function
Each agent has a Cobb-Douglas utility function, normalized to be on a scale from $0$ to $1$. The utility derived by agent $i$ from neighborhood $k$ is given by 
$$
U_{i,k}=q_{i,k}^{\theta} \cdot c_{i,k}^{1-\theta}
$$
where:
- $q_{i,k}=$ neighborhood quality for agent $i$ in neighborhood $k$, given by the proportion of agents in $k$ with an income bracket $\geq$ that of agent $i$.
- $c_{i,k}=$ proportion of income left for consumption on other goods besides housing, given by $\dfrac{{Y_{i}-\text{Rent}_{k}}}{Y_{i}}$.
- $\theta=$ how much the agent values neighborhood quality relative to consumption. A $\theta$ closer to $1$ would behave more like a Schelling agent, giving more preference to the composition of the neighborhood, while $\theta$ closer to $0$ makes the agent behave more like a textbook economic agent. By default, $\theta \sim \text{Uniform}(0.6,0.8)$.

## Housing logic
Houses are initialized at a fixed price, given by `STARTING_HOUSE_PRICE` in `common.py`. Supply of houses is always fixed to be equal to the number of agents, so that any homelessness is caused purely due to demand side factors. 

We have made an important assumption here: **all houses in a certain neighborhood are perfect substitutes.** We ignore granular details that could distinguish housing units and simplify it to focus mainly on the segregation aspect of the model. This means *we've given every house in the neighborhood the same price for simplicity.* 

### Price update rule
We update the price of every house in a neighborhood using the reservation price so that supply = demand and the market clears at the new price. The reservation price is given by the **lowest winning bid in the neighborhood.**

If demand exceeds supply, house prices are set to the reservation price. If supply exceeds demand, we look at two factors: the minimum sustainable price (MSP) and the `decay_rate`. We multiply all house prices by the `decay_rate` so that prices fall a bit if demand is low. MSP is the WTP of the poorest agent in the neighborhood when their utility is zero (essentially $\beta Y_{i}$). It acts like a *price floor* to prevent the decay rate from pulling prices down forever. This is especially useful when the model is close to equilibrium, since few agents move and the decay rate (in the absence of MSP) would keep depreciating the prices.

Finally, we clip the prices with `max_change` to prevent extreme volatility in pricing between rounds, and we compare the clipped price to the MSP to make sure the price floor isn't violated.

### Pricing out agents
As rents go up due to demand, it is possible that an agent previously residing in that neighborhood can no longer afford to do so. We define an agent's max WTP for their current neighborhood $j$ as 
$$
B^{\text{stay}}_{i,j}=\text{min}((\beta+\gamma U_{i,j})Y_{i}, \delta Y_{i})
$$
This formula is identical to the bidding formula, so an agent's WTP is considered to be *how much he would've bid for this neighborhood.* If $B^{\text{stay}}_{i,j}<\text{Rent}_{j}$, then the agent's bid would not have cleared the cutoff in the auction phase, so he would not have secured a home here. Therefore, he is evicted since he can no longer afford neighborhood $j$. 

Since the evicted agent is homeless now (and thus unhappy), he is eligible to bid in the same round itself since evicting priced out happens before the bidding in a round. This also frees up his old home for a new higher bidder to bid on.

#### What happens to agents without a home?
If an agent is priced out of a home/cannot win any bid to get a home, he is moved into the *nonmarket housing category*. The main reason we do this is to keep agents in a given neighborhood so that their incomes are considered for other agents happiness checks and the average neighborhood income (for visualization). The accumulation of these poorer agents who can't afford market housing will push out richer agents, effectively endogenizing slum creation into the model.

## Main algorithm
Our main algorithm is:
`happiness check` $\to$ `evict priced out` $\to$ `bidding` $\to$ `allocate houses` $\to$ `update prices` $\to$ `happiness check`.

Our sim ends in three cases:
1) All agents are happy
2) If the `converge` flag is `True`, the sim converges if churn (% of agents who move) for the last `convergence_bound` rounds is zero.
3) If the `converge` flag is `False` (default), then the sim ends after `max_rounds` iterations are complete ($100$ by default).

## Optimizations used
This sim is very computationally heavy, with a time complexity of `O(r × (n × k + n log n))`, simplifiying to `O(r x k x n)`, meaning it scales for `r` rounds, each scanning all `n` agents/houses in all `k` neighborhoods, with the default parameter values `r=100, k=100, n=1_000_000`. Since the number of operations is approximately given by $T(r, n, k) \;=\; O\big(r \times (n \times k + n \log n)\big)$, this means we use around 12 billion operations per sim.

To make the sim runnable in a reasonable time on consumer hardware, I've used many optimizations.

Firstly, **I've made extensive use of `numba` and JIT compilation.** Using the `cache = True` and `parallel = True` flags wherever possible allowed me to cache the compiled functions (in `/src/__pycache__`) to reduce compilation time and to parallelize functions, making better use of available CPU threads.

Next, **I used `numpy` to vectorize functions to the greatest extent possible.** Vectorizing everything allows Python to process multiple items in an array at once, instead of looping through each element and applying the operation one by one as you'd do in vanilla Python. With millions of operations per iteration, increased speed of doing operations becomes crucial, making vectorization a key optimization in my model.

Lastly, **I used datatype declaration to reduce memory consumption by $\sim 50\%$**. Every structured array I use to store data has explicitly declared datatypes to save on memory, since `float32/int8` is enough to store almost everything in my sim, making the usual $64$ bit allocations unnecessary. This should prevent the large arrays from exploding the memory usage for sims with a large number of agents.

### Benchmarks
I own a ASUS Zenbook 14 OLED laptop with $16$ GB RAM and an AMD Ryzen 7 8840HS CPU. While plugged in, I got the following results:

1) 1 mil agents, $100$ rounds = 3m 49.2s
	- Without `numba`, this took $\sim 106$ mins for 25 rounds, so I'd assume it takes 424 mins for $100$ rounds. **Therefore, `numba` alone gives us a $\sim 99\%$ speedup.**

## Stats
`stats.py` gives us a way to track many stats for each round of our sim and visualize them, to get more of a look into what our sim is doing. The following stats are tracked for each round:
~~~python
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
	("prev_house", np.int32, (num_agents,)), # used to calc churn
	("churn", np.float32),
	("num_bids", np.uint32),
	("winning_bids", np.uint32)
])
~~~

Note: `dissimilarity` will be calculated soon once I figure out the logic behind it.

## Parameters

`N_AGENTS` - How many agents/houses are in our sim 

`N_NEIGHBORHOODS` - Number of neighborhoods
`GRID_SIZE` - Side of the square when visualizing the neighborhoods (have to implement this better)

`PERCENTILES` - Gives us the upper and lower bound of each bracket. The length of this array minus one gives us `N_BRACKETS`, which is the number of income brackets we're working with

`DEFAULT_HAPPINESS_PERCENT` - Tells us what % of agents in the neighborhood must have >= income bracket compared to our agent for him to be happy

`STARTING_HOUSE_PRICE` - The price at which we initialize the houses.

`DECAY_RATE` - Multiplier to get new house price when supply > demand (new price = old price * decay) 
`MAX_CHANGE` - Maximum percent change in one round, meant to prevent large price swings (between 0-1)

`BETA` - The baseline fraction of income an agent is willing to pay for a house 
`GAMMA` - The marginal WTP for one unit of utility
`DELTA` - Maximum percent of income an agent can spend on rent (from 0-1)

`THETA_MIN, THETA_MAX` - The minimum and maximum values of an agent's preference for social utility over consumption

## License
This project is licensed under the [MIT License](./LICENSE)