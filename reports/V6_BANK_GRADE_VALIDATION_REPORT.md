# V6 Bank-Grade Deployment & IFRS 9 Validation Report

## Conclusion

V6 receives a conditional pass for portfolio demonstration. The build adds outcome-maturity controls, separate LGD and EAD proxy models, IFRS 9 staging and ECL, probability-weighted macroeconomic scenarios, controlled deployment gates, incident management and rollback evidence.

## Delayed outcomes

The monitoring process contains 17 monthly cohorts, of which four have at least 80% mature 12-month outcomes at the reporting date. AUC, KS, observed bad rate and calibration are withheld for immature cohorts rather than presented as zero or as successful results. Data quality, score drift, approval behavior and early-warning indicators provide interim evidence.

## IFRS 9

Accounts are assigned to Stage 1, Stage 2 or Stage 3 using default status, days past due, relative PD deterioration and watchlist indicators. Stage 1 uses 12-month PD; Stage 2 uses lifetime PD; Stage 3 uses a PD of 100%. ECL combines modeled PD, LGD and EAD and is discounted at 10%. Upside, Base and Downside ECL results are weighted 20%, 50% and 30%.

## Deployment decision

Shadow and 5% canary gates pass. The 25% canary gate fails because p95 latency, error rate and decision-reconciliation measures exceed limits. Full rollout is blocked. The rollback drill passes all five controls and restores `RCS_PD_4.0` with `RETAIL_POLICY_1.0`.

## Limitations

- Application, default, recovery and macroeconomic data are synthetic.
- SICR, cure, default and write-off definitions are illustrative and are not an approved accounting policy.
- LGD and EAD targets are simulated proxies rather than observed recovery and utilization histories.
- Macroeconomic multipliers are scenario sensitivities rather than econometrically estimated satellite models.
- Production deployment requires managed infrastructure, real-time observability, security testing, independent validation and committee approval.
