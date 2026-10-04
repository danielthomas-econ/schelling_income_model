# Measurement

[← Documentation index](README.md)

This page covers how the model's outcomes are measured:
- the statistics recorded every round;
- how Monte Carlo runs are combined, and how the policy is compared with the baseline;
- how welfare is measured, household by household.

## Statistics recorded every round

`get_stats` and `get_mc_stats` in `src/stats.py` record these after every round:

| Statistic | Definition |
|---|---|
| `happiness` | % of households happy ($q \ge 0.5$ in a market home) |
| `nonmarket_housing` | % of households in nonmarket housing |
| `vacancies` | % of homes empty. Equals `nonmarket_housing` in the baseline, since homes = households |
| `avg_value` | mean market rent across all homes |
| `avg_income` | mean resident income of each neighborhood (drives the segregation GIF) |
| `theil_between`, `theil_within` | the Theil decomposition below |
| `theil` | mean of the 100 neighborhoods' own Theil indices |
| `gini` | mean of the neighborhoods' own Gini coefficients, ×100 |
| `churn` | % of households whose home changed since the last round, including moves into or out of nonmarket housing |
| `num_bids`, `winning_bids` | bids placed and vacancies filled this round |

All neighborhood statistics use **residents only**: households renting a market home there. A household in nonmarket
housing doesn't count toward the neighborhood it left.

### Segregation: the Theil decomposition

The Theil T index of resident incomes splits exactly into a part *within* neighborhoods and a part *between* them.
With $N$ residents of mean income $\mu$, and neighborhood $k$ holding $N_k$ residents of mean income $\mu_k$ and its
own Theil index $T_k$:

$$T = \frac{1}{N}\sum_i \frac{y_i}{\mu}\ln\frac{y_i}{\mu}
= \underbrace{\sum_k \frac{N_k \mu_k}{N\mu}\, T_k}_{\text{within}}
+ \underbrace{\sum_k \frac{N_k}{N}\,\frac{\mu_k}{\mu}\ln\frac{\mu_k}{\mu}}_{\text{between}}$$

The within term weights each neighborhood by its share of total **income**; that weighting is what makes the identity
exact. The **between share**, between ÷ $T$, is the segregation measure the README reports: the fraction of
city-wide income inequality explained by which neighborhood a household lives in. It is 0 in a perfectly mixed city
and 1 in a fully stratified one.

The Gini column is complementary: it averages inequality *inside* each neighborhood. As neighborhoods homogenize it
falls, and a policy that mixes incomes raises it.

**A caution for policy comparisons.** Because these statistics count residents only, a policy that pushes some
households into nonmarket housing changes *who* is being measured, not just how they are sorted. A change in the
between share under the policy mixes both effects.

## Monte Carlo design

**Independent cities.** `monte_carlo_sim` repeats the simulation `n_runs` times. Run $r$ is seeded with `seed + r` and
draws a **fresh city**: new incomes, new $\theta$s, and new random placement of homes across neighborhoods. Run-to-run
variation therefore includes differences between cities, not just randomness in households' choices.

**Confidence intervals.** For any statistic, the mean across runs gets a 95% $t$-interval,
$\bar x \pm t_{0.975,\,n-1}\, s/\sqrt{n}$. `mc_mean_ci` does this round by round, over the runs that reached that
round. `band = "spread"` instead shows the middle 95% of individual runs, which describes run-to-run variability
rather than uncertainty about the mean.

**Paired policy comparison.** `evaluate_policy` runs the baseline Monte Carlo and the policy Monte Carlo with the
**same seed**. Since the policy's setup makes the same random draws in the same order
([policy](policy.md#the-rule)), baseline run $r$ and policy run $r$ start from the identical city, and household $i$
is the same household in both. A check that the total income matches in each pair guards this. Effects are then
computed as **paired differences**: for each run, policy minus baseline at the final round, with a $t$-interval over
the runs (`paired_difference`). Most of the city-to-city noise cancels within each pair, so the intervals are much
tighter than comparing two independent sets of runs would give. The two runs follow different paths once the policy
starts changing the market, so household choices don't stay synchronized; only the starting city is shared.

## Welfare: equivalent variation

### Each household's end state

At the end of each run, `welfare_state` records for every household its neighborhood quality $q$, its budget share
left after rent $c$, and its utility $u = q^\theta c^{1-\theta}$. A household whose current home is worse than
nonmarket housing would leave next round, so it is assigned nonmarket housing's values ($q = q_{nm}$, $c = 1$).

### From utility to money

Utility levels can't be added up or compared across households: the scale of $q^\theta c^{1-\theta}$ is arbitrary,
and $\theta$ differs from household to household. Instead, each household's change is converted into money with
**equivalent variation (EV)**. For each household, find the budget share $c_{eq}$ that, in its *baseline*
neighborhood, would give it its *policy* utility:

$$q_{\text{base}}^{\theta}\, c_{eq}^{1-\theta} = u_{\text{policy}}
\quad\Longrightarrow\quad
c_{eq} = \left(\frac{u_{\text{policy}}}{q_{\text{base}}^{\theta}}\right)^{\frac{1}{1-\theta}},
\qquad
\text{EV} = \delta\,(c_{eq} - c_{\text{base}})$$

(`equivalent_variation` in `src/policy.py`). EV is a share of income. An EV of $-0.03$ means the policy hurts the
household as much as a rent increase of 3% of its income would in its baseline situation; $+0.03$ is worth a 3%-of-
income rent cut. Measuring in shares of income rather than rupees means every household counts equally in an
average, instead of the rich dominating it.

This is EV *in place*: the compensation is computed holding the household in its baseline neighborhood, without
letting it re-sort in response to the money. A textbook EV would let it re-optimize, including moving. The in-place
version is simpler and keeps the measure local, but it can overstate the money needed to match a large change in
neighborhood quality.

### Aggregation

- **The median EV**, overall and within each income bracket: the headline welfare number, robust to outliers.
- **The shares better and worse off:** EV above $+10^{-4}$ or below $-10^{-4}$ percentage points, which treats
  floating-point noise as no change.
- **A capped mean.** EV can be extreme for high-$\theta$ households: matching a large change in neighborhood quality
  with money alone enters to the power $1/(1-\theta)$, which is 10 at $\theta = 0.9$. Means therefore cap each
  household's EV at ±65% of income (its whole housing budget), so a handful of such households can't dominate. Medians
  and shares don't need the cap.
- **By outcome group:** households are split by where they end up under the policy (eligible in a reserved home,
  eligible in a market home, eligible in nonmarket housing, ineligible and housed, ineligible in nonmarket housing).
  This shows *who* drives the averages.

Every statistic is computed per run, and the reported value is the mean across runs with a 95% $t$-interval.

**Comparing preference specifications.** EV under status and EV under homophily come from different utility functions
(different $q$), so their *signs and patterns* are comparable but their exact magnitudes are not.
