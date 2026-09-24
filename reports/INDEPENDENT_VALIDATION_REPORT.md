# Independent Validation Report — RCS_PD_4.0

## Validation opinion

The model is approved for portfolio demonstration subject to the limitations below. Approval does not authorize use in real credit decisions because development and validation use synthetic data.

## Scope

The review challenges conceptual soundness, data and feature controls, discrimination, probability calibration, stability, decision strategy, implementation consistency and ongoing monitoring.

## Findings

1. The monotonic WOE logistic scorecard meets the minimum out-of-time AUC and KS thresholds.
2. Sigmoid calibration materially improves the interpretation of PD relative to the class-weighted raw estimator. Mean predicted PD is compared with the observed default rate.
3. Out-of-time PSI remains below the escalation threshold of 0.25 under controlled population drift.
4. The random-forest challenger provides a non-linear benchmark. Champion selection retains the calibrated logistic model because it balances performance, stability and explainability.
5. WOE and IV outputs support variable challenge. They are analytical diagnostics; the production estimator remains a calibrated logistic pipeline.
6. Policy cut-offs are separated from model estimation and assessed through approval rate, observed bad rate, expected loss and expected profit.
7. API output is traceable through model ID, calibrated PD, score, decision, reason codes and expected loss.
8. Accepted-only and fuzzy-augmentation tests illustrate reject-inference sensitivity. Synthetic full-information results provide a benchmark unavailable in a real rejected population.
9. Fairness monitoring covers age band, employment type and acquisition channel. Segment differences require investigation and cannot be interpreted as proof of discrimination without legal and causal review.
10. Behavioural early warning assigns Green, Amber and Red statuses using utilisation, payment, delinquency, cash-flow and bureau signals.
11. Development, validation and out-of-time windows are separated chronologically. Vintage monitoring reports FPD30, MOB3 30+ and 12-month bad rate.
12. Cut-off economics, pricing and limit assignment use RCS_PD_4.0 scores consistently.
13. Production controls include an immutable audit schema, version registry, automated performance gates and a rollback mechanism. PostgreSQL and CI/CD deployment remain demonstrative until connected to an institution's infrastructure.

## Required controls before real use

- Replace synthetic records with governed, representative and time-stamped institutional data.
- Confirm target definition, cure rules, observation window, exclusions and leakage controls.
- Perform fairness, privacy, legal, fraud, KYC/AML and affordability reviews.
- Complete independent code replication, UAT, security testing, access control and disaster recovery testing.
- Obtain Model Risk Committee and delegated-credit-authority approval.
- Establish outcome maturation, override monitoring and monthly data-quality controls.

## Monitoring triggers

| Indicator | Amber | Red / action |
|---|---:|---:|
| Score PSI | 0.10–0.25 | ≥0.25; investigate and assess recalibration |
| Gini decline | 10%–20% from benchmark | >20%; restrict use and assess redevelopment |
| Calibration gap | 2–3 percentage points | >3 percentage points; recalibrate |
| Missing input rate | 1%–3% | >3%; investigate upstream data |
| Override rate | 5%–10% | >10%; review policy and governance |
