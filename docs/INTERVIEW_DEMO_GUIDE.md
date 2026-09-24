# Seven-minute Interview Demo

## 1. Business problem — 45 seconds

Explain that the platform supports unsecured retail application decisions while separating model risk estimation from credit policy and human authority.

## 2. Application scoring — 75 seconds

Enter one application. Show PD, score, point-based reason codes, decision, interest rate, limit, expected loss, profit, RAROC, model ID, policy version and audit ID.

## 3. Model development and validation — 90 seconds

Show monotonic WOE bins, chronological development/validation/OOT windows, AUC, Gini, KS, Brier score and calibration. Explain why an explainable scorecard may be selected even when a challenger has slightly higher discrimination.

## 4. Strategy and portfolio impact — 75 seconds

Change the approval cut-off. Explain the movement in approval rate, bad rate, expected loss and profit. Show pricing, limit assignment and vintage monitoring.

## 5. Governance — 75 seconds

Show version registry, CI/CD gates, manual overrides, role separation, audit trail and rollback. State clearly that committee approval and real production release are outside the synthetic portfolio demonstration.

## 6. Closing — 30 seconds

Summarize the lifecycle: develop, independently validate, approve, deploy, monitor, remediate and retire.

## Questions to expect

1. Why did you select WOE logistic regression instead of the model with the highest AUC?
2. How do you prove that score points reproduce model PD?
3. How do you handle rejected applicants?
4. How do model version and policy version differ?
5. What triggers recalibration or redevelopment?
6. How are overrides controlled and monitored?
7. What changes before using real customer data?
# V5 five-minute demonstration

1. Open **V5 Summary** and explain why the champion remains in place despite testing a more complex challenger.
2. Open **Walk Forward** and show that performance is measured on four chronological unseen periods, not a random split alone.
3. Open **Strategy Simulator**, change the approval and review PD cut-offs, and explain the trade-off between approvals, bad rate, expected profit and RAROC.
4. Open **Stress Test** and compare Base, Downturn and Severe outcomes.
5. Close with the deployment boundary: synthetic data, candidate policy, independent validation, committee approval and controlled pilot are still required.

Suggested interview statement: “I treated model selection and credit strategy as separate governance decisions. The challenger did not clear the pre-defined uplift hurdle, while the candidate cut-offs were optimized under credit-quality and profitability constraints and then tested under stress.”

## V6 interview demonstration

1. Explain why recent cohorts cannot yet produce valid 12-month AUC or observed bad-rate conclusions.
2. Show the mature-label rate and the leading indicators used during the outcome delay.
3. Walk through Stage 1, Stage 2 and Stage 3 ECL and the probability-weighted macroeconomic scenarios.
4. Show the failed 25% canary gate, incident escalation and five completed rollback controls.
5. Conclude that a safe rollback is evidence of effective production governance, not a failed project.

Suggested statement: “I designed monitoring around the availability of the target outcome. Before labels mature, the platform relies on data quality, drift, decision and early-delinquency indicators. Once outcomes mature, discrimination and calibration metrics are released. Deployment uses shadow and canary gates, and the simulated 25% rollout demonstrates that the process can stop and recover safely.”
