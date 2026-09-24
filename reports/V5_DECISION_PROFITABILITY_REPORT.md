# V5 Decision & Profitability Validation Report

## Executive conclusion

V5 passes the portfolio simulation gates. The WOE logistic scorecard remains the champion. Its mean AUC across four chronological validation folds is 0.679, compared with 0.669 for the gradient-boosting challenger. The challenger therefore does not meet the pre-defined 0.02 AUC-uplift hurdle.

The constrained strategy simulation recommends an approval PD cut-off of 13% and a review cut-off of 17% for further validation. On the synthetic out-of-time population, this combination produces a 57.5% approval rate, a 7.8% observed bad rate, expected profit of approximately VND 8.78 billion and portfolio RAROC of 126.3%.

## Stress results

The scenario framework shocks PD odds, LGD and funding cost. Expected profit remains positive in Base, Downturn and Severe scenarios for the tested synthetic portfolio. This is a screening result, not evidence of resilience on a real bank portfolio.

## Governance decision

The active production-simulation policy remains `RETAIL_POLICY_1.0`. The optimized cut-offs are stored separately as `RETAIL_POLICY_2.0_CANDIDATE`. They must not become active until the model and decision strategy are independently validated on governed bank data, approved by the Credit Committee and tested through a controlled pilot with monitoring and rollback criteria.

## Material limitations

- Customer and performance data are synthetic.
- LGD, CCF, funding cost, operating cost and economic-capital assumptions are illustrative.
- The profitability model does not include prepayment, cure timing, collections cash flows, tax, liquidity premiums or lifetime ECL.
- The stress scenarios are deterministic sensitivities rather than statistically estimated macroeconomic satellite models.
- Production identity, managed database, concurrent-load testing and penetration testing remain outside this portfolio implementation.
