# Parameters

[← Documentation index](README.md)

Model constants live in `src/common.py`. Most can be overridden per run as keyword arguments to `sim_one_round`,
`monte_carlo_sim`, `evaluate_policy` and the policy versions; the **Argument** column gives the name.

## Population and city

| Parameter | Default | Argument | Meaning |
|---|---|---|---|
| `N_AGENTS` | 1,000,000 | `n_agents` | households, and homes (one home per household) |
| `N_NEIGHBORHOODS` | 100 | (fixed) | neighborhoods; see [implementation](implementation.md#known-limitations-of-the-code) |
| `PERCENTILES` | 0, 10, ..., 90, 95, 99, 100 | | bracket cutoffs: deciles, then 90-95th, 95-99th and top 1% |

## Preferences

| Parameter | Default | Argument | Meaning |
|---|---|---|---|
| `PREFERENCE` | `"status"` | `preference` | what counts as a similar neighbor: `"status"` (bracket at or above one's own) or `"homophily"` (within the window) |
| `HOMOPHILY_WINDOW` | 1 | `homophily_window` | brackets either side of one's own that count as similar under homophily |
| `THETA_MIN`, `THETA_MAX` | 0.1, 0.9 | `theta_min`, `theta_max` | range of the weight on neighbors vs money, $\theta \sim U(\text{min}, \text{max})$ |
| `DELTA` | 0.65 | `delta` | share of income available for housing; no household pays more than this in rent |
| `DEFAULT_HAPPINESS_PERCENT` | 0.5 | `happiness_percent` | Schelling threshold: households with fewer similar neighbors than this look to move |
| `NONMARKET_QUALITY` | 0.1 | `nonmarket_quality` | nonmarket housing quality as a fraction of a random mix: $q_{nm}(b) = 0.1\,s(b)$ |
| `CHOICE_TEMPERATURE` | 0.1 | `temperature` | logit choice noise; smaller means closer to always picking the best neighborhood |
| `COUNT_NONMARKET` | `False` | (constant) | whether households in nonmarket housing still count toward the neighborhood they left |

## Housing market

| Parameter | Default | Argument | Meaning |
|---|---|---|---|
| `MAX_CHANGE` | 0.10 | (constant) | largest change in a neighborhood's rent in one round, up or down |
| `RESERVATION_RENT_SHARE` | 0.05 | (constant) | landlords' reservation rent, as a share of the city's median income (≈ ₹18,800/yr) |
| `STARTING_HOUSE_PRICE` | ₹1,00,000 | `starting_house_price` | no effect: round 0's auctions set the first rents |

## Policy

These are arguments of `evaluate_policy` and the `_affordable` sim functions.

| Argument | Default | Meaning |
|---|---|---|
| `income_cutoff` | 2 | brackets 0 to this are eligible (bottom 30%) |
| `houses_eligible` | 0.2 | share of each neighborhood's homes reserved |
| `lower_price` | 0.6 | rent on a reserved home, as a share of the market rent |

## Runs

| Argument | Default | Meaning |
|---|---|---|
| `max_rounds` | 100 | rounds per run |
| `n_runs` | 30 (`monte_carlo_sim`, `evaluate_policy`) | Monte Carlo runs; the README uses 20 |
| `seed` | `None` (0 in `evaluate_policy`) | makes runs reproducible; run $r$ uses `seed + r` |
| `redraw_population` | `True` | draw a new city for every Monte Carlo run |
| `converge`, `convergence_bound` | `False`, 5 | stop early once no household has moved for this many rounds |
| `plot` | `True` | show plots and printed summaries |

`parameter_sweep` varies one of `delta`, `nonmarket_quality`, `temperature`, `theta_min` or `theta_max` over a grid
and records the end-of-run statistics for each value.
