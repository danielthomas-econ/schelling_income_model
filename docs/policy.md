# The set-aside policy

[← Documentation index](README.md)

The policy reserves a fixed share of every neighborhood's homes for low-income households at a discount. It runs on
the same market as the baseline ([The housing market](housing-market.md)), with changes at three points: who may bid
where, how vacancies are allocated, and what counts toward the market rent. The code is in `src/policy.py`.

## The rule

| Setting | Default | Code |
|---|---|---|
| Eligible households | brackets 0-2 (bottom 30% of incomes) | `income_cutoff = 2` |
| Reserved homes | 20% of the homes **in every neighborhood** | `houses_eligible = 0.2` |
| Rent on a reserved home | 60% of the neighborhood's market rent | `lower_price = 0.6` |

- **Reserved homes are flagged once, at the start** (`assign_low_rent_house`): in each neighborhood, the first
  $\lfloor 0.2\,H \rfloor$ homes are reserved. The quota is uniform: it doesn't depend on where eligible households
  want to live.
- **Every home keeps a market rent** (`houses["value"]`), set by the market exactly as in the baseline. A reserved
  home charges its tenant `rent_charged` $= 0.6 \times$ market rent (`set_rent_charged`), so the discount moves with
  the market.
- **Only eligible households can rent a reserved home.** A reserved home that no eligible household takes stays empty.

**Same city, with and without the policy.** `new_population_affordable` makes exactly the same random draws, in the
same order, as the baseline's `new_population`. With the same seed, a policy run starts from the identical city:
same incomes, same $\theta$s, same homes in the same neighborhoods. Any difference in outcomes is the policy's doing.
[Measurement](measurement.md#monte-carlo-design) explains how this pairing is used.

## What changes in each step of a round

`run_round_affordable` follows the same five steps as `run_round`. Steps 1-2 (neighborhood quality, happiness) are
unchanged.

**3. Evictions.** Tenants are judged against the rent they actually pay. A reserved-home tenant pays 60% of the market
rent, so it can stay at market rents well above its stay value.

**4. Bidding.** The two groups face different rents and options:

| | Sees which rent? | Can bid where? |
|---|---|---|
| **Eligible** | the discounted rent where a reserved home is vacant, the market rent elsewhere | any neighborhood with a vacant home of either kind |
| **Not eligible** | the market rent | only neighborhoods with a vacant *market* home |

Each household still picks one neighborhood by logit choice and bids its bid-rent.

**5. Allocation: two auctions per neighborhood** (`allocate_houses_affordable`):

1. **Reserved homes.** Eligible bidders are taken in descending order of bid. Each fills a vacant reserved home if
   its bid covers the discounted rent, and pays the discounted rent. Because all reserved homes in a neighborhood
   share one price, the first eligible bid that falls short ends this auction.
2. **Market rent signal.** The clearing auction ([details](housing-market.md#the-rent-signal-a-uniform-price-clearing-auction))
   runs over the **market homes only**: market tenants' stay values against the bids of everyone still unplaced
   (ineligible bidders, and eligible ones who didn't get a reserved home).
3. **Market homes.** The remaining bidders fill vacant market homes at the market rent, exactly as in the baseline.

Running the reserved auction first, restricted to eligible households, is what makes the policy targeted.
[Design decisions](design-decisions.md#the-set-aside-needs-its-own-auction) explains why a single auction failed.

**Rent update.** The market rent moves by the same rule as in the baseline, with one difference: **vacant reserved
homes are not counted as vacancies** (`update_prices` excludes them). They aren't market supply, so they can't pull
the market rent down. Reserved rents are then reset to 60% of the new market rent.

## Why the policy backfires: the mechanism

The README's headline result comes from a chain of four links. Each one follows from the rules above.

1. **Reserved homes empty out where eligible households can't or won't live.** As the city sorts, reserved homes in
   the richer neighborhoods lose their eligible tenants and no eligible household replaces them. By the end, about 40%
   of reserved homes stand empty under status and about 70% under homophily
   ([why](#why-reserved-homes-stand-empty)).
2. **Empty reserved homes are withdrawn supply.** No one else may rent them, and they don't count as vacancies. The
   market therefore loses every unfilled reserved home: more market vacancies are absorbed, and neighborhoods that
   used to have spare homes now have excess demand.
3. **Excess demand raises market rents.** With more claims than market homes, clearing rents rise (by up to 10% a
   round) until enough households drop out. In neighborhoods that used to have spare homes, the rent jumps from the
   reservation-rent floor to whatever the marginal bidder will pay. This is why the size of the effect depends on $R$.
4. **Higher rents price households out.** As rents pass tenants' stay values, more households end up in nonmarket
   housing, including eligible households who didn't get a reserved home and now face higher market rents.

The winners are the eligible households who land a reserved home: they pay 40% less. Even for them, the discount isn't
always enough. A reserved home can be in a neighborhood where the household has fewer similar neighbors than it had in
the baseline (under homophily, a poor household in a mixed neighborhood), and the cheaper rent may not make up for
it. Reserved homes also bring poorer neighbors into richer neighborhoods, which under status preferences lowers $q$
for everyone above them.

## Why reserved homes stand empty

The numbers in this section come from one seeded run (100k households, seed 0, 100 rounds). The Monte Carlo averages
in the README tell the same story.

### They are filled first, then empty out as the city sorts

Share of reserved homes vacant, by round:

| Round | 0-5 | 10 | 20 | 40 | 99 |
|---|---|---|---|---|---|
| Status | 0% | 0% | 13% | 40% | 42% |
| Homophily | 0% | 2% | 67% | 68% | 68% |

In round 0 every neighborhood is an unsorted mix with low rents, and every reserved home finds an eligible tenant.
The vacancies appear between rounds 10 and 40, while the city sorts. Then, in neighborhood after neighborhood:

1. **Richer households move in**, and the market rent rises. The discounted rent is 60% of the market rent, so it
   rises with it.
2. **The neighborhood's mix moves away from the eligible tenant's own bracket.** This lowers the tenant's $q$, and
   with it the most it would pay to stay.
3. **The tenant is priced out** once the discounted rent passes that stay value
   ([evictions](housing-market.md#3-evictions)). The reserved home becomes vacant.
4. **No eligible household takes it back**, for the reasons below. It stays empty for the rest of the run.

### By the end, they are concentrated in the richer neighborhoods

Neighborhoods grouped into fifths by the mean bracket of their residents, at round 99:

| Neighborhoods, poorest to richest | 1st fifth | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| **Status:** discounted rent (₹/yr) | 15k | 45k | 98k | 170k | 259k |
| **Status:** reserved homes vacant | 3% | 0% | 16% | 92% | 100% |
| **Homophily:** discounted rent (₹/yr) | 63k | 86k | 91k | 103k | 133k |
| **Homophily:** reserved homes vacant | 0% | 45% | 95% | 100% | 100% |

The two preferences empty these homes for different reasons:

- **Status: the discount isn't enough.** Under status the poorest bracket values every neighborhood fully ($q = 1$),
  so the obstacle is money. An eligible household's whole housing budget (65% of income) is at most about ₹1.15
  lakh a year for the poorest bracket, and ₹1.73 lakh for the richest eligible one. Its bid-rent is always below that
  budget, so a discounted rent of ₹1.7-2.6 lakh in the richest fifth of neighborhoods is out of reach at any
  preference.
- **Homophily: the neighborhoods aren't worth living in.** The discounted rents are lower, but a poor household's $q$
  in a middle- or upper-income neighborhood is close to zero. Its bid-rent there is near zero, so it prefers free
  nonmarket housing even at a 40% discount.

### The homes that would be taken are in the wrong places

Eligible households don't lack demand for reserved homes; it's in the wrong neighborhoods. Under homophily, about 9,000
of the 30,000 eligible households end the run in nonmarket housing. Yet the reserved homes in the poorest fifth of
neighborhoods, the only ones they want, are all occupied. The quota is fixed at 20% of *every* neighborhood, so the
supply of reserved homes doesn't follow eligible households' demand. Some neighborhoods have too few reserved homes and
others too many.

### Households don't search for cheaper rent

The model has one more reason, specific to how households search. A household only looks for a new home when it is
unhappy with its neighbors (the [Schelling trigger](households.md#happiness-the-schelling-trigger)). A cheaper rent
somewhere else doesn't, on its own, prompt it to look. At round 99, checking every empty reserved home against every
eligible household:

| | Status | Homophily |
|---|---|---|
| Empty reserved homes | 8,348 | 13,504 |
| ...that some eligible household would be strictly better off in | 2,082 | 304 |
| ...where that household is actually searching | 0 | 0 |

Most empty reserved homes have no eligible household that would be better off in them. For the few that do, the
households that would gain are content where they live, so they never look. With price-driven search, those homes
would fill, but they are too few to change the picture: the vacancies come mainly from the first two reasons.

## What the model leaves out

Housing supply is fixed. In practice, set-asides are usually attached to *new*
construction, and their main effects run through what developers choose to build. The model captures the narrower
question of what a uniform set-aside does to an existing stock.
