# Model Card — RCS_PD_4.0

## Intended use

Application-stage ranking and decision support for unsecured retail lending. The monotonic WOE logistic scorecard estimates default within 12 months and returns point-based reason codes. The platform also demonstrates time-based validation, pricing, limit assignment, overrides, audit logging and lifecycle governance. It does not replace affordability, fraud, KYC/AML, eligibility, or human-review controls.

## Data and target

- Development sample: 8,000 synthetic applications
- Holdout sample: 4,000 synthetic applications
- Out-of-time monitoring sample: 3,000 synthetic applications with controlled drift
- Target: `default_12m` (1 = default within 12 months)
- Features: demographics limited to age, affordability, employment stability, bureau history, requested facility, tenor and acquisition channel

## Approval criteria

| Test | Minimum / limit | Outcome source |
|---|---:|---|
| Holdout AUC | ≥ 0.65 | `model_metadata.json` |
| Holdout KS | ≥ 0.25 | `model_metadata.json` |
| OOT score PSI | < 0.25 | `model_metadata.json` |
| Calibration gap | < 3 percentage points | `model_metadata.json` |
| Reproducible scoring | Required | automated smoke test |

## Monitoring and escalation

- Green: PSI < 0.10; continue monthly monitoring.
- Amber: 0.10 ≤ PSI < 0.25; investigate feature and channel drift.
- Red: PSI ≥ 0.25; escalate to Model Risk Committee and assess recalibration/redevelopment.
- Monitor monthly approval rate, review rate, decline rate and input completeness.
- Monitor quarterly observed bad rate, AUC/Gini, KS and calibration by score band once outcomes mature.
- Apply champion/challenger testing before replacing the production version.

## Limitations

This is a portfolio implementation built with synthetic data. It requires institution-specific data lineage, representativeness testing, legal/fair-lending review, independent model validation, UAT, security review and formal approval before real lending use.
# V5 decisioning extension

- Four chronological walk-forward validation folds test performance through time.
- The gradient-boosting challenger must improve mean AUC by at least 0.02 before it proceeds to full independent validation.
- The decision layer estimates EAD, expected loss, break-even rate, expected profit, economic capital and RAROC.
- Base, Downturn and Severe scenarios shock PD odds, LGD and funding cost.
- Strategy recommendations are analytical outputs only; credit committee approval and validation on real bank data remain required.

# V6 deployment and IFRS 9 extension

- Performance metrics based on observed defaults are reported only for cohorts with sufficiently mature 12-month outcomes.
- Separate synthetic LGD and EAD proxy models are validated with MAE and R-squared before use in ECL calculations.
- IFRS 9 outputs include Stage 1, Stage 2 and Stage 3, lifetime PD, discounted ECL and probability-weighted macroeconomic scenarios.
- A controlled rollout simulation covers shadow scoring, 5% and 25% canary traffic, incident escalation, rollout blocking and rollback evidence.
- V6 has conditional approval only because all customer, recovery and macroeconomic data remain synthetic.
