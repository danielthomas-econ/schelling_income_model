# Households

[← Documentation index](README.md)

Everything a household does in the model, from bidding to leaving its home to choosing among neighborhoods, comes
from one utility function and one closed-form consequence of it, the bid-rent. This page builds both up from the
income data.

## Incomes

Incomes are drawn from Delhi's household income distribution. `data/income_quantile_delhi.csv` is a quantile
function (10,001 points, from ₹114,000 at the bottom to ₹3.59M at the top) approximated from CMIE's Consumer Pyramids
household survey. To draw incomes (`get_incomes` in `src/agents.py`):

1. Fit a smoothing spline to **log** income against the quantile (`UnivariateSpline`, smoothing `s = 5`). The log
   scale and the smoothing remove the survey's step-like jumps without distorting the shape of the distribution.
2. Draw $u \sim U(0, 1)$ for each household and set $y = \exp(\text{spline}(u))$ (inverse-transform sampling).

The survey caps the top of the distribution, so the richest households in the model are richer than most but not
"billionaire rich". That matters most for the top bracket, which every status-seeking household wants to live near.

## Income brackets

Households are grouped into 12 brackets by percentile of the *simulated* incomes (`find_income_brackets`), so every
bracket holds exactly its intended share of the city:

| Bracket | 0-8 | 9 | 10 | 11 |
|---|---|---|---|---|
| Percentiles | each decile (0-10th, ..., 80-90th) | 90-95th | 95-99th | top 1% |
| Share of households | 10% each | 5% | 4% | 1% |

Brackets, not raw incomes, are what households perceive in their neighbors.

## Neighborhood quality $q$

A household in bracket $b$ judges a neighborhood by the share of its residents who are in a bracket *similar* to
$b$. Two definitions of "similar" are implemented (`PREFERENCE` in `src/common.py`), each as a range of brackets
$[\ell_b, h_b]$ (`similar_brackets` in `src/agents.py`):

- **Status:** brackets at or above one's own, $[b, 11]$. Everyone wants neighbors at least as rich as they are.
- **Homophily:** brackets within one of one's own, $[b-1, b+1]$ (clipped at 0 and 11; the width is
  `HOMOPHILY_WINDOW`). Everyone wants neighbors like themselves.

$$q_k(b) = \frac{\#\{\text{residents of } k \text{ with bracket in } [\ell_b, h_b]\}}{\#\{\text{residents of } k\}}$$

Only households renting a market home count as residents. Households in nonmarket housing don't count toward the
neighborhood they left (`COUNT_NONMARKET = False`).

The two definitions differ in who is hurt by mixing. Under **status**, a poorer neighbor lowers $q$ for everyone
above them, while a richer neighbor never lowers anyone's $q$. So mixing is costly for almost everyone, and the
poorest bracket is indifferent to composition: $q = 1$ everywhere. Under **homophily**, mixing is symmetric: a richer
neighbor lowers $q$ for the poor just as a poorer neighbor lowers it for the rich.

**The city-wide mix, $s(b)$.** The share of the whole city similar to bracket $b$ is what $q$ would be in a perfectly
mixed neighborhood (`city_quality`):

| Bracket | 0 | 1 | 2-7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| Status $s(b)$ | 100% | 90% | 80%-30% | 20% | 10% | 5% | 1% |
| Homophily $s(b)$ | 20% | 30% | 30% | 25% | 19% | 10% | 5% |

It is used in two places: as the expected $q$ of an **empty** neighborhood (every neighborhood is empty in round 0,
and households assume a random mix will move in), and to set the quality of nonmarket housing (below).

## Utility

A household in bracket $b$ living in neighborhood $k$ at rent $r$ gets

$$U = q_k(b)^{\theta}\, c^{1-\theta}, \qquad c = \frac{\delta y - r}{\delta y}$$

(`utility` in `src/common.py`).

- **$\delta y$ is the housing budget.** $\delta = 0.65$ of income is available for housing; the rest is committed to
  other necessities. $c$ is the share of that budget left after rent: 1 when housing is free, 0 when rent uses the
  whole budget. Rents at or above $\delta y$ give $U = 0$, so no household ever pays more than 65% of its income.
- **$\theta$ is the weight on neighbors versus money**, drawn per household from $U(0.1, 0.9)$. A household with
  $\theta$ near 0 barely cares who its neighbors are; one near 1 would spend almost anything to live among similar
  households. The ends are trimmed because a home's only attribute in the model is its neighborhood: a household
  with $\theta = 0$ would never prefer a market home to free nonmarket housing.
- **Utility is scale-free in income.** Because $c$ is a *share* of the budget, a rich and a poor household with the
  same $\theta$ face the same trade-off between neighbors and money. Income enters through what each can bid in
  rupees, not through the utility level.

## The outside option: nonmarket housing

A household that can't get or keep a market home lives in **nonmarket housing**: free ($c = 1$), but of low quality.
It stands in for informal housing. Its quality for bracket $b$ is a discounted random mix:

$$q_{nm}(b) = 0.1 \times s(b)$$

where 0.1 is `NONMARKET_QUALITY`,

so its utility is $U_{nm} = q_{nm}(b)^{\theta}$.

Why scale by $s(b)$ instead of using one constant for everyone: $q$ is measured relative to the household's own
bracket, so under status a perfectly mixed neighborhood gives the top 1% a $q$ of only 0.01. A flat $q_{nm}$ of,
say, 0.2 would make every mixed neighborhood worse than nonmarket housing for the top brackets, which would then
never bid and never cluster.

Households in nonmarket housing are always unhappy, so they keep looking for a market home every round.

## The bid-rent

The **bid-rent** $r^*$ is the highest rent at which a neighborhood is still at least as good as nonmarket housing.
Setting the two utilities equal and solving for rent:

$$q^{\theta} c^{1-\theta} = q_{nm}^{\theta}
\quad\Longrightarrow\quad
r^* = \delta y \left(1 - \left(\frac{q_{nm}}{q}\right)^{\frac{\theta}{1-\theta}}\right)$$

and $r^* = 0$ when $q \le q_{nm}$ (`bid_rent` in `src/common.py`). Its properties drive most of the model:

- **It rises with $q$.** Better neighbors are worth more rent.
- **It is proportional to income.** At the same $q$ and $\theta$, a household with twice the income bids twice as
  much. This is why the rich win the neighborhoods everyone wants.
- **It rises with $\theta$** (for $q > q_{nm}$): households that care more about neighbors pay more for them.
- **It never exceeds the housing budget $\delta y$.**

For example, a household with income ₹3 lakh and $\theta = 0.5$, looking at a neighborhood where 60% of residents are
similar ($q = 0.6$) against nonmarket quality $q_{nm} = 0.03$, has
$r^* = 0.65 \times 300{,}000 \times (1 - 0.05) \approx ₹185{,}000$ a year. With $\theta = 0.2$ the same household would
pay only about ₹103,000.

The same function is used three ways, so bidding, evictions and the auctions all follow one rule:

1. **Bids.** A household bids its bid-rent for the neighborhood it picks ([bidding](housing-market.md#4-bidding)).
2. **Stay values.** A tenant's bid-rent for its *current* neighborhood is the most it would pay to stay. If the
   rent rises above it, the tenant leaves for nonmarket housing ([evictions](housing-market.md#3-evictions)).
3. **Auction claims.** Tenants' stay values compete with newcomers' bids to set the neighborhood's rent
   ([auctions](housing-market.md#5-auctions-and-rents)).

## Happiness: the Schelling trigger

A household is **happy** if it rents a market home and at least half its neighbors are similar:
$q_k(b) \ge 0.5$ (`check_happiness`, `DEFAULT_HAPPINESS_PERCENT`). Happy households stay put and don't bid. Unhappy
households, including everyone in nonmarket housing, look for somewhere better.

Happiness only decides *who searches*. *Where* they go and how much they pay come from utility and the bid-rent. One
consequence: under status preferences the poorest bracket always has $q = 1$, so its housed members are always
happy and never move.
