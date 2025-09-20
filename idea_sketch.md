## agent behavior
assign agent income based on `income_quantile.csv`, which is a quantile function of india's income distribution derived from CMIE data (done ✅)

divide agents into different income brackets. we have deciles until the highest tiers, where it becomes 90-95%ile, 95-99%ile, 99%ile+ (done ✅)

give each agent a location in a 10x10 grid. each 1x1 square is called a neighborhood, assign each one a number (done ✅)

agents want at least x% of their neighbors (other agents living in the neighborhood) to be of an equal or higher income category than them (done ✅). currently testing to see the effects of agents wanting x% of their neighbors to have +-1 income bracket as them.

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
we update the house prices for all units in the neighborhood using the *reservation price*.

the price of houses in the neighborhood is determined by supply (vacant houses) and demand (number of bids). if demand > supply, house price = lowest winning bid (market clearance price at which supply = demand). if supply > demand, the house prices will decay by a small factor `decay_rate`.

house prices are also clipped so that they dont exceed `max_change`% in a round

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
\> cool idea: use scikit-optimize to find the optimal value of all my greek letter parameters
\> **VERY INTERESTING IDEA:** currently, segregation is fuelled by both preference for similar income neighbors and market constraints. try to break down this effect and find out what each effect's contribution to segregation is (like how price effect is decomposed into income and substitution effect). maybe use shapley decomposition?
how to do it:
## Shapley Decomposition Advantages

**1. Handles Non-Additive Interactions**
Shapley values properly account for all possible interaction effects between your two forces without assuming they're simply additive.

**2. Unique Solution**
Unlike ad hoc decompositions, Shapley values give you the unique "fair" attribution of each factor's contribution to total segregation.

**3. Axiomatic Foundation**
Shapley decomposition satisfies desirable properties:
- **Efficiency**: All segregation is attributed to some factor
- **Symmetry**: Identical factors get identical attributions  
- **Dummy**: Factors with no impact get zero attribution
- **Additivity**: Works even with complex interactions

## Implementation for Your Model

Run these four simulations:

```python
def shapley_segregation_decomposition():
    # All possible combinations of your two factors
    S_none = run_simulation(equal_income=True, no_preferences=True)      # Baseline
    S_market = run_simulation(real_income=True, no_preferences=True)     # Market only
    S_pref = run_simulation(equal_income=True, preferences=True)         # Preferences only  
    S_both = run_simulation(real_income=True, preferences=True)          # Full model
    
    # Marginal contributions in both orders
    market_contribution_first = (S_market - S_none) + (S_both - S_pref)
    market_contribution_second = (S_both - S_pref) + (S_market - S_none)
    
    pref_contribution_first = (S_pref - S_none) + (S_both - S_market)
    pref_contribution_second = (S_both - S_market) + (S_pref - S_none)
    
    # Shapley values (average marginal contributions)
    shapley_market = (market_contribution_first + market_contribution_second) / 2
    shapley_preference = (pref_contribution_first + pref_contribution_second) / 2
    
    return shapley_market, shapley_preference
```

## Simple Explanation

Think of Shapley decomposition as asking: **"If I add this factor to any possible combination of other factors, what's its average marginal contribution?"**

For your case:
- What does income inequality add when there are no social preferences?
- What does income inequality add when social preferences already exist?
- Average these to get the "fair" attribution to income inequality

Same logic for social preferences.

## Why Shapley is Better for Your Research

**1. Robustness**: Works regardless of interaction complexity
**2. Interpretability**: Each factor gets a single, well-defined contribution value
**3. Academic Credibility**: Standard method in applied economics (used in growth accounting, wage gap decomposition, etc.)
**4. Policy Relevance**: Tells you which factor to target for maximum segregation reduction

## Research Contribution

Using Shapley decomposition for segregation analysis would be genuinely novel - most urban economics papers use simple counterfactuals. This methodological contribution alone could make your paper publishable.

The Shapley approach is mathematically rigorous and handles the complexity of your two-factor system properly, whereas the simple additive decomposition I initially suggested makes unrealistic assumptions about how market and preference forces interact.
------------------------------
\> analyze through the lens of a willingness-to-pay (WTP) premium for social utility (moving to a preferred neighborhood). we can change $\theta$ and $\lambda$ to range from 'economic' agents (pure rational) for low values to 'sociological' agents (derive greater utilty from neighborhood)
\> scatter plot showing correlation between income bracket and avg income of current neighborhood
\> look at government intervention and conduct rcts to test the impact

### concerning things
\>  Correlation with Bidding Success: Analyze the correlation between an agent's c_term and their success rate in winning bids for socially desirable neighborhoods (i.e., neighborhoods where q_diff is high). You should find that agents with lower c_term values (due to high current rent burdens) are less likely to win bids, even for neighborhoods that would make them happy.

The Constraint: Because most agents desire higher-income neighbors, and there are inherently few high-income individuals, it becomes a competitive, zero-sum game for these desirable neighbors. Not everyone can achieve their high x% threshold for "better" neighbors, leading to the observed 1-x happiness convergence. The "proportion of the population that can realistically achieve this preference" is limited by the availability of the desired income brackets.

### to do before next commit
\> scrap the lognormal distribution and base the distribution off a cdf from a real world dataset
current dataset is CMIE income pyramid (household level)


### More things we can implement?
#### Segregation measures to compute (and what they capture)

Dissimilarity index (D): share of a group that would need to move to achieve even distribution. Simple and widely used in urban segregation literature. (Used in Indian neighborhood papers.) 
paa2019.populationassociation.org
CEGA

Isolation index (or exposure): probability that a member of group A meets another member of A in their neighborhood — useful to measure clustering of poor vs rich. 
paa2019.populationassociation.org

Gini / Theil of neighborhood mean incomes: a measure of economic segregation across space (how unequal neighborhood average incomes are). 
CEGA

Moran’s I and LISA (Local Indicators of Spatial Association): spatial autocorrelation measures — these tell you whether high incomes cluster in space and where the hotspots are. Good when you have geographies (shapefiles).

Comparisons over multiple scales: compute indices at enumeration-block, ward, and city levels — segregation often looks very different at different spatial scales. 
SSRN