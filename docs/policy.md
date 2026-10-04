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

1. **Reserved homes land where eligible households can't or won't live.** A reserved home is only taken if an
   eligible household's bid-rent covers the discounted rent.
   - In a rich neighborhood, even 60% of the market rent can exceed a poor household's entire housing budget.
   - Under homophily, a poor household's $q$ in a rich neighborhood is close to zero, so its bid-rent there is zero
     at any price.
   - So the uniform 20% quota leaves many reserved homes empty: most of them under homophily, fewer under status,
     where the poorest bracket values every neighborhood equally ($q = 1$).
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

**What the model leaves out.** Housing supply is fixed. In practice, set-asides are usually attached to *new*
construction, and their main effects run through what developers choose to build. The model captures the narrower
question of what a uniform set-aside does to an existing stock.
