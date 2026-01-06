### future plans
\> cool idea: use scikit-optimize to find the optimal value of all my greek letter parameters
\> implement a TUI dashboard for the project for the aesthetics
\> **VERY INTERESTING IDEA:** currently, segregation is fuelled by both preference for similar income neighbors and market constraints. try to break down this effect and find out what each effect's contribution to segregation is (like how price effect is decomposed into income and substitution effect). maybe use shapley decomposition?
how to do it:
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

## to do before next commit
\> look at Bruch & Mare 2006; Benenson 1999; Zhang 2004


- check why some runs lead to 0 happy in certain income brackets