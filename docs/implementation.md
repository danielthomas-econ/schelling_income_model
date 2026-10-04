# Implementation

[← Documentation index](README.md)

How the model is built to run a million households on a laptop, stay reproducible, and be checked.

## Code map

| File | Role |
|---|---|
| `src/common.py` | settings; `utility` and `bid_rent` (the whole preference model); seeding; stopping rule |
| `src/agents.py` | income draws, brackets, neighborhood composition $q$, happiness |
| `src/bidding.py` | which neighborhoods each household can bid on, the logit choice, the bids |
| `src/houses.py` | stay values, evictions, the per-neighborhood auctions, rent updates, reservation rent |
| `src/sim.py` | `run_round`, single runs, Monte Carlo, parameter sweeps |
| `src/policy.py` | the set-aside versions of the above, welfare measurement, `evaluate_policy` |
| `src/stats.py` | per-round statistics and confidence intervals |
| `src/plots.py` | all plotting; the simulation modules only compute |
| `src/debug.py` | the same loops with per-step timings, for profiling |
| `scripts/make_readme_results.py` | regenerates every number and figure in the README |

## Data layout

Households and homes are each one NumPy **structured array**, one row per household or home, with explicit dtypes:

- **`agents`:** `id`, `income`, `income_bracket` (uint8), `neighborhood` (int8), `happy`, `house` (index into
  `houses`, −1 in nonmarket housing), `nonmarket_housing`, `rent_paid`, `theta` (float32), plus `low_rent` (eligibility)
  under the policy.
- **`houses`:** `id`, `tenant` (index into `agents`, −1 if vacant), `neighborhood` (uint8), `value` (market rent),
  plus `rent_charged` and `low_rent` under the policy.

One contiguous array per entity, rather than a list of separate arrays, cuts memory by roughly 80% for the agent table
and lets Numba-compiled loops read whole rows at once.

The link between the two arrays (each household's `house` and each home's `tenant`) is the model's main invariant.
`agent_house_mapping` restores it after every change: a household whose home no longer lists it as tenant is moved to
nonmarket housing, and a home whose listed tenant lives elsewhere is marked vacant.

## Performance

The inner loops are compiled with **Numba** (`@njit(cache = True)`; compiled code is cached between sessions).
Loops whose iterations are independent run in parallel with `prange`: neighborhood composition, happiness, and stay
values. The bidding kernel runs serially because it draws random numbers. Aggregations such as composition counts use
vectorized NumPy (`np.add.at`, `np.bincount`).

One million households for 100 rounds takes about 3.5 minutes on a laptop (Ryzen 7 8840HS). An early pure-Python
version took about 106 minutes for 25 rounds. `src/debug.py` reports the time of each step of a round, for profiling.

**Known hotspot.** The per-neighborhood auctions and rent updates scan the full home array once per neighborhood, so
that part of a round scales with homes × neighborhoods. Sorting homes by neighborhood once at the start would remove
that factor.

## Reproducibility

`set_seed(seed)` seeds **two** random number generators: NumPy's, which is used for incomes, $\theta$ and home
placement, and Numba's, which keeps a separate state and drives households' logit choices.

They must not receive the same seed. Both use the Mersenne Twister, so identical seeds give identical streams: the
uniform draws that generate incomes would reappear as the draws households use to choose neighborhoods, and round 0
came out perfectly sorted by income. `set_seed` therefore derives two independent seeds with
`np.random.SeedSequence(seed)`.

With a seed, a single run, a Monte Carlo (run $r$ uses `seed + r`) and the full README pipeline are all
deterministic. Rerunning `scripts/make_readme_results.py` reproduces its tables exactly.

## Verification

Checks run while building the current version:

- **Refactors reproduce old results exactly.** After plotting was split out and the preference logic generalized,
  seeded runs (baseline and policy, single run and Monte Carlo) matched the previous outputs bit for bit across 96
  arrays.
- **Neighborhood composition matches a brute-force count**, for homophily windows 0-2.
- **The Theil decomposition sums to the city-wide index** to about $10^{-16}$.
- **Household and home bookkeeping stays consistent.** Households without a home, vacant homes, and households in
  nonmarket housing match exactly (a check in `plots.ipynb`).
- **Every entry point runs end to end:** single runs, Monte Carlo, policy evaluation, parameter sweeps, debug runs and
  plots.

These checks were run by hand; there is no automated test suite yet. Turning the bookkeeping invariant and the
Theil identity into tests is a natural next step.

## Known limitations of the code

- **The neighborhood count is fixed.** `N_NEIGHBORHOODS` (100) is used directly inside the compiled composition
  functions, so the `n_neighborhoods` argument of the sim functions only sizes the statistics arrays. The plotting
  functions also assume a 10×10 grid, and household neighborhoods are stored as int8, which caps the count at 127.
- **The policy duplicates the simulation loops.** `src/policy.py` repeats the single-run and Monte Carlo loops with
  the policy's steps swapped in, so a change to one loop has to be mirrored in the other.
