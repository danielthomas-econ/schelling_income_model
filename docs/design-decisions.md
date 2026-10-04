# Design decisions

[← Documentation index](README.md)

Most rules in the model replaced an earlier version that produced a wrong or fragile result. This page records why each
rule is the way it is, and what went wrong before. It's the reasoning behind the mechanisms in the other pages, and
a guide to what not to undo.

## Preferences and choice

### One function for bids, evictions and auction claims
The bid-rent $r^*$ is the only link between preferences and money. A newcomer bids it, a tenant leaves when its rent
passes it, and tenants' values in it set the clearing rent. Earlier versions used two ad hoc rules: a bid of
$\min((\beta + \gamma U)\,y,\ \delta y)$, scaled by utility $U$, and a stay limit of
$\min((\beta + \gamma q)\,y,\ \delta y)$, scaled by neighborhood quality $q$ alone. Because they disagreed, a
household could win a home at a rent above what it would then pay to stay. Deriving all three from
$U = q^\theta c^{1-\theta}$ makes them consistent by construction, and removes the two free parameters $\beta$ and
$\gamma$.

### Rent enters the utility of the neighborhood being considered
Early versions computed the budget term $c$ from the rent a household was *currently* paying, not the rent of the
neighborhood it was considering. Utility comparisons across neighborhoods were then meaningless. $c$ now always uses
the rent of the neighborhood being evaluated.

### Logit choice instead of always picking the best neighborhood
Households in the same bracket see the same neighborhood qualities and rents, so if each picked its single best
option, a whole bracket would pile onto the same neighborhood in the same round while near-identical alternatives
stayed empty. Logit choice ($\tau = 0.1$) keeps the best option most likely but spreads households across close
alternatives.

### Nonmarket housing quality depends on the bracket
A single outside-option quality for everyone (say 0.2) would make every mixed neighborhood worse than nonmarket
housing for the top brackets, whose $q$ in a random mix is tiny under status (0.01 for the top 1%). They would never
bid and never cluster. Scaling nonmarket
quality by each bracket's city-wide share, $q_{nm}(b) = 0.1\,s(b)$, keeps the outside option equally unattractive
relative to a random mix for every bracket.

### Brackets are cut from the simulated incomes
Bracket cutoffs used to come from a different fit of the income data than the one used to draw incomes. The brackets
were then not true deciles: in testing the bottom bracket held about 12% of households, and the top-1% bracket could
come out empty. Cutoffs are now percentiles of the drawn incomes themselves.

## The housing market

### Tenants are part of the rent auction
Rents used to be set only by the bids of newcomers for vacancies. Two failures followed:
- **Rents crashed in rich neighborhoods.** A few poorer outsiders winning a couple of vacancies could set a low
  price for a rich neighborhood.
- **Rents spiked from single bids.** One rich outsider bidding on an otherwise full neighborhood could push
  everyone's rent up 10% in a round.

With logit choice spreading bids across the city, this evicted most of it. The clearing auction now pools tenants'
stay values with newcomers' bids, so the rent rises only when outsiders would outbid the marginal tenant.

### One rent per neighborhood
Winners used to pay their own winning bid while their neighbors paid the prevailing rent, which was often higher.
A new arrival could then be priced out the very next round. Now every home in a neighborhood rents for the same price,
winners pay that price, and only the rent update moves it.

### Moving out vacates the old home
A household that won a home elsewhere used to stay listed as the tenant of its old home as well, so every move
silently destroyed a home ("ghost tenancies"). Winners now vacate their old home first, and `agent_house_mapping`
enforces one home per household and one household per home.

### Full neighborhoods hold their rent
The rent used to fall 5% whenever a neighborhood had no excess demand, including when it was full. Households only
bid where there is a vacancy, so a full neighborhood never received outside bids. Its rent fell every round, and
rents ended up at 0.2-2% of income. A full neighborhood with no excess demand now keeps its rent.

### Rents fall with the vacancy share, down to a reservation rent
The flat 5% cut for any vacancy had two problems:
- **It was a knife-edge.** One empty home out of a thousand cut the rent as much as five hundred.
- **There was no floor.** Where vacant homes were unwanted at any rent, rents decayed geometrically for 100 rounds
  ($0.95^{100} \approx 0.6\%$) to about ₹1,000 a year, while full neighborhoods stayed near ₹90,000-180,000. The
  policy comparison then partly measured the policy ending that artifact.

The current rule cuts the rent by the vacancy share and floors it at the landlords' reservation rent.

**Alternatives considered:**
- **The old "minimum sustainable price"** (0.3 × the poorest resident's income). It made the floor depend on *who
  lived there* rather than on the cost of supplying housing, and could cascade: evicting the poorest resident raised
  the floor, which could evict the next. It was removed along with the bidding rule it belonged to.
- **Letting rents fall only as far as unhoused demand justifies.** This needs no reservation rent, but where nobody
  wants a home at any positive rent, the competitive rent is the cost of supply, so a floor is still needed. A simple
  version also stepped down far too slowly where demand was dense, leaving thousands unhoused next to hundreds of
  empty homes.
- **A natural-vacancy-rate rule**, $\Delta r / r = \lambda(v^* - v)$. With exactly one home per household, every
  natural vacancy is a household in nonmarket housing, so $v^*$ would effectively fix the nonmarket share by
  assumption.

### The set-aside needs its own auction
The first policy version reused the baseline auction, which ignored which homes were reserved and who was eligible.
Reserved homes went to the highest bidders, who were mostly richer households, and anyone who won one got the
discount. Now reserved vacancies are filled first, from eligible bidders only, and the market clearing rent is computed
over market homes only, so discounted homes can't drag down the market rent. Reserved homes also start at their
discounted rent from round 0; at first they charged the full price until the first rent update.

## Measurement

### Each Monte Carlo run draws a new city
Every run used to start from a copy of the same city, so the only variation across runs was the randomness in
households' choices. The confidence intervals described one city, not the model, and were far too narrow. Each run
now draws fresh incomes, $\theta$s and home placements.

### Separate seeds for NumPy and Numba
Seeding both generators with the same number made them produce the same stream, so the uniform draws that created
incomes reappeared as households' choice draws, and round 0 came out perfectly sorted by income. Two independent seeds
are now derived from one with `SeedSequence` ([implementation](implementation.md#reproducibility)).

### Rounds with a genuine zero aren't dropped
Monte Carlo averages used to discard every value equal to zero, on the assumption that 0 meant "this run had already
ended". But churn, nonmarket housing and bid counts can genuinely be zero, so those rounds silently vanished. Each run
now records the last round it reached, and averages use exactly the runs that reached each round.

### The Theil decomposition is exact
The within-neighborhood term used population weights rather than income weights, so within + between didn't add up to
the city-wide Theil, and the between share came out 0.2-0.8 percentage points too high. Both terms now use the
weights under which the decomposition is exact ([measurement](measurement.md#segregation-the-theil-decomposition)).
