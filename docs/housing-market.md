# The housing market

[← Documentation index](README.md)

The housing market is what turns preferences into sorting. Without prices, every household would simply move to
wherever its neighbors are most similar. With prices, households compete for the neighborhoods they want, and income
decides who wins. This page follows one round step by step (`run_round` in `src/sim.py`).

## The housing stock

- **One home per household.** There are exactly $N$ homes for $N$ households, so any household outside market
  housing is there because of prices and preferences, never because of a physical shortage.
- **Homes are spread at random across 100 neighborhoods** (`initialize_houses`), so each neighborhood has about
  $N/100$ homes, give or take a few percent.
- **One rent per neighborhood.** Every market home in a neighborhood rents for the same price $r_k$
  (`houses["value"]`). New arrivals and long-standing tenants pay the same.

**Starting state.** Every household begins in nonmarket housing and every home is empty. Rents start undefined: in
round 0 every neighborhood is empty, so the first auction discovers its rent
([below](#a-completely-empty-neighborhood)). The starting price `STARTING_HOUSE_PRICE` has no effect on results.

## 1. Neighborhood quality

For every neighborhood $k$ and bracket $b$, compute $q_k(b)$, the share of residents similar to $b$
(`neighborhood_quality` in `src/agents.py`; see [Households](households.md#neighborhood-quality-q)). An empty
neighborhood is assigned the city-wide mix $s(b)$. Nonmarket housing quality $q_{nm}(b) = 0.1\,s(b)$ is computed at
the same time.

All later steps in the round use this snapshot. Households don't anticipate how this round's moves will change the
neighborhoods.

## 2. Happiness

Housed households with $q \ge 0.5$ are happy and sit this round out. Everyone else is unhappy and may bid
(`check_happiness`).

## 3. Evictions

Each tenant's **stay value** is its bid-rent for its current neighborhood: the most it would pay to stay rather than
move to nonmarket housing (`stay_values` in `src/houses.py`). A tenant whose rent is above its stay value leaves
for nonmarket housing, and its home becomes vacant (`check_priced_out`, `evict_priced_out`).

A tenant can be pushed out in two ways:
- **The rent rises** above what it will pay.
- **Its neighbors change.** If similar neighbors leave, $q$ falls, so its stay value falls too.

The second channel is what makes sorting self-reinforcing.

## 4. Bidding

Each unhappy household chooses **one** neighborhood to bid for (`_place_bid_kernel` in `src/bidding.py`).
Neighborhood $k$ is an option only if all three hold:

1. **It has a vacant home** (`vacant_neighborhoods`).
2. **Its current rent is below the household's bid-rent**, $r_k < r^*_{ik}$, i.e. at today's rent it beats nonmarket
   housing. An empty neighborhood shows a rent of 0, since its rent hasn't been set yet.
3. **For a household that already has a home, it is strictly better than that home**: $U_k > U_{\text{current}}$ at
   today's rents. Housed households only move up.

Among its options, the household picks one by **logit choice**:

$$P(k) \propto \exp\!\left(\frac{U_k / U_{\text{best}} - 1}{\tau}\right), \qquad \tau = 0.1$$

where $\tau$ is `CHOICE_TEMPERATURE`.

The best option is the most likely, but an option with 90% of the best utility still gets weight $e^{-1}$ relative to
it. Without this, every household in a bracket would pile onto the same best neighborhood in the same round, and
rents would whipsaw. Smaller $\tau$ means closer to always picking the best.

The household then **bids its bid-rent** $r^*_{ik}$ for the chosen neighborhood. That is the most it would pay;
whether it actually pays that much is decided by the auction.

## 5. Auctions and rents

Each neighborhood runs its own auction (`allocate_houses` in `src/houses.py`), which does two jobs: it decides who
moves into the vacant homes, and it produces the signal that moves the rent.

### The rent signal: a uniform-price clearing auction

All claims on the neighborhood's $H$ homes are pooled:
- every current tenant's **stay value** (its claim to keep its home), and
- every **outside bid** from households that chose this neighborhood and don't already live there.

If there are more claims than homes, the **clearing rent** is the $(H+1)$-th highest claim: the highest claim that
doesn't get a home (`clearing_price`). If claims don't exceed homes, there is no excess demand and the signal is 0.

Pooling tenants with newcomers matters. If only newcomers set the price, a few poor outsiders winning a couple of
vacancies could crash the rent of a rich neighborhood, and a single rich outsider could push everyone's rent up.
With tenants in the pool, a neighborhood's rent rises only when outsiders would pay more than the tenants with the
lowest stay values, which is exactly when those tenants are about to be priced out. That makes the auction and the
eviction rule consistent.

### Who moves in

Vacant homes go to the highest outside bids that are at least the neighborhood's current rent. Winners **pay the
current rent**, not their own bid, so the whole neighborhood keeps one price. A winner who already had a home
elsewhere gives it up, and that home becomes vacant.

### A completely empty neighborhood

A neighborhood with no tenants (every neighborhood in round 0) has no rent yet, so its auction discovers one
(`vacancy_price`):
- **more bidders than homes:** the highest losing bid;
- **otherwise:** the lowest bid, so everyone who bid gets a home.

That rent is floored at the reservation rent $R$, and it becomes the neighborhood's rent.

### The rent update

After the auctions, each neighborhood's rent moves (`update_prices`):

| Situation | New rent before the cap |
|---|---|
| **Excess demand**: clearing rent > 0 | the clearing rent |
| **Vacant market homes**, no excess demand | $r \times (1 - v)$, where $v$ is the share of the neighborhood's market homes that are vacant |
| **Full**, no excess demand | unchanged |

Two limits then apply:
- **A ±10% cap per round** (`MAX_CHANGE`), so rents move smoothly instead of jumping.
- **The reservation rent floor:** no rent goes below
  $R = 0.05 \times$ median income, about ₹18,800 a year (`RESERVATION_RENT_SHARE`).

**Why rent falls with the vacancy share.** One empty home out of a thousand is weak evidence that the rent is too
high; a third of the homes standing empty is strong evidence. So the rent falls in proportion: 1 empty home in
1,000 cuts it by 0.1% a round, and 10% vacancy (or more) cuts it by the full 10%.

**Why a floor.** Some vacant homes are unwanted at *any* rent: under homophily, a poor household values a
middle-income neighborhood less than nonmarket housing, so it won't bid even at zero. Without a floor, the rent in
such a neighborhood keeps falling toward zero, and its residents end up living almost for free. The reservation rent
is the landlords' cost of supplying a home (maintenance, taxes, the value of the next-best use): below it, they
won't let the home. Wherever homes are in excess supply, the rent settles at $R$, so $R$ affects the magnitude of
some results. The README reports results for $R$ at 2%, 5% and 10% of median income.

The new rent applies from the next round. A household that moved in this round may find the rent rising above its
stay value later and leave; the 10% cap keeps this from happening en masse.

## How sorting emerges

The steps above add up to a feedback loop:

1. **Higher incomes bid more.** Bid-rents are proportional to income, so in any neighborhood many households want,
   the richer bidders win the vacancies.
2. **Competition raises the rent.** When outside bids outnumber vacancies, the clearing rent rises toward what the
   bidders will pay.
3. **The marginal tenants leave.** Tenants whose stay values the new rent passes (lower incomes, or lower $\theta$)
   move out.
4. **The neighborhood's mix shifts up**, changing $q$ for everyone: it rises for the rich, which raises their bids
   further, and falls for those who no longer fit, which lowers their stay values.

The loop repeats until neighborhoods are stratified enough that most households are happy or can't afford a better
option. In the default runs this takes about 30-40 rounds; the model runs 100 rounds by default (`max_rounds`), with
an optional stop once no household has moved for 5 consecutive rounds (`converge = True`, `should_stop` in
`src/common.py`).
