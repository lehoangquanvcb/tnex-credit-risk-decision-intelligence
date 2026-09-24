# Retail Credit Risk Platform V9 — TNEX Production Credit Risk & Decision Intelligence

An end-to-end portfolio project demonstrating monotonic WOE scorecard development, walk-forward validation, credit strategy optimization, separately validated PD–LGD–EAD components, IFRS 9 ECL, delayed-outcome monitoring, shadow/canary rollout, incident response and controlled rollback. All customer records are synthetic and safe for demonstration.

V7 adds a TNEX-aligned, configurable digital-lending layer for Cash Loan, BNPL and Business Loan: fraud–credit dual decisioning, alternative-data governance, sub-five-second orchestration, digital funnel analytics, repeat-customer limit management and payment/collections reconciliation. It is a portfolio simulation and is not an official TNEX model or affiliated implementation.

V8 turns that layer into an interactive operating simulation: a live single-application decision API, separate product calibration evidence, linked-device fraud intelligence, joint risk–profit strategy optimization, collections next-best-action and a compact command center.

V9 is the final functional edition. It trains separate Cash Loan, BNPL and Business Loan pipelines, allocates a simulated VND 1 trillion portfolio under concentration and loss limits, manages a model inventory and validation findings, monitors customer outcomes, and prepares an executive Risk Committee pack.

## Business objective

Estimate 12-month probability of default at application, translate PD into a customer score, and apply transparent policy cut-offs:

- **APPROVE:** PD < 8%
- **REVIEW:** 8% ≤ PD < 18%
- **DECLINE:** PD ≥ 18%

## Architecture

`Synthetic application data → data controls & feature pipeline → logistic PD model → holdout/OOT validation → decision rules → API/Streamlit → PSI & performance monitoring`

## Validation framework

- Discrimination: ROC-AUC, Gini and KS; champion–challenger comparison
- Probability calibration: sigmoid calibration, Brier score and predicted-versus-observed PD
- Explainability: production WOE logistic scorecard, point contribution and adverse-action reason codes
- Selection bias: accepted-only and fuzzy-augmentation reject-inference comparison
- Responsible lending: approval, bad-rate and disparate-impact monitoring by segment
- Customer lifecycle: behavioural PD and Green/Amber/Red early-warning actions
- Governance: development, validation, UAT, committee, release and monitoring workflow
- Production controls: PostgreSQL audit schema, local audit fallback, Docker Compose, model registry, rollback and CI/CD approval gates
- Credit strategy: V4-aligned cut-offs, risk-based pricing, limit assignment, RAROC and manual override queue
- Validation design: development, validation and out-of-time windows plus vintage delinquency monitoring
- Release consistency: calibration, fairness, reject inference and monitoring stamped with the same model, dataset and build identifiers
- Hardening controls: scorecard reconstruction, bin-level stability, versioned policy, JWT/RBAC, override approval, audit search and latency gates
- V5 challenger governance: four chronological walk-forward folds and an explicit minimum 2 percentage-point AUC uplift rule before challenger validation
- V5 portfolio economics: EAD, expected loss, break-even pricing, expected profit, economic capital and RAROC by risk band
- V5 decision strategy: joint approval/review cut-off simulation with profitability and observed bad-rate constraints
- V5 stress testing: Base, Downturn and Severe scenarios covering PD odds, LGD and funding-cost shocks
- V6 delayed-outcome monitoring: mature-label controls, interim leading indicators, PSI and performance metrics released only when the observation window is complete
- V6 IFRS 9: Stage 1/2/3 allocation, 12-month and lifetime PD, modeled LGD/EAD, discounted ECL and probability-weighted macroeconomic scenarios
- V6 deployment controls: shadow scoring, 5% and 25% canary gates, incident register, automated rollout block and evidence-based rollback drill
- V7 TNEX-aligned product configuration: separate limits, credit cut-offs, fraud thresholds and policy versions for Cash Loan, BNPL and Business Loan
- V7 real-time decisioning: eKYC, alternative-data features, fraud scoring, credit scoring, policy and offer orchestration with a five-second end-to-end SLA
- V7 customer lifecycle: application funnel, repeat-loan limit actions, payment reconciliation, reminder suppression and hardship/complaint routing
- V8 live decision simulator: product PD, fraud score, network signals, affordability, limit, reason codes, model ID and policy version in one auditable response
- V8 product models: separate Cash Loan, BNPL and Business Loan calibration evidence and release status
- V8 fraud network: linked-device clusters with step-up verification or block-and-investigate actions
- V8 risk–profit optimizer: approval, expected credit loss, fraud loss, funding cost, operating cost, contribution and RAROC across policy combinations
- V8 collections next-best-action: digital reminder, promise-to-pay, recovery review, hardship routing and mandatory contact suppression for unreconciled payments
- V9 independent product pipelines: separate preprocessing, training, out-of-time testing, model artifacts and validation decisions for each lending product
- V9 portfolio risk appetite: product allocation under concentration, bad-rate, ECL, contribution and RAROC constraints
- V9 model governance: materiality, accountable owner, independent validator, deployment stage, next review and tracked findings
- V9 customer outcomes: affordability, debt burden, repeat borrowing, complaint and hardship indicators by product and income segment
- V9 Risk Committee pack: concise decisions, status, owner and follow-up actions for senior management
- Calibration proxy: Brier score and decile-level predicted vs. observed default rate
- Stability: out-of-time AUC, bad-rate movement and score PSI
- Governance: versioned model ID, metadata, validation status and documented cut-offs
- Production controls: typed API inputs, health endpoint, reproducible pipeline and smoke test

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/modeling.py
python -m src.v5_decisioning
python -m src.v6_bank_grade
python -m src.v7_tnex_decisioning
python -m src.v8_digital_lending_os
python -m src.v9_production_intelligence
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs` after starting Uvicorn.

## Portfolio language

> Built and hardened an end-to-end retail credit risk platform using Python, including a monotonic WOE scorecard, walk-forward champion–challenger validation, separately validated LGD and EAD models, probability-weighted IFRS 9 ECL, delayed-outcome monitoring, shadow/canary deployment, incident management, rollback controls, JWT/RBAC, PostgreSQL audit design and CI/CD gates.

This sentence should be presented as a **portfolio project**, not as production experience at a former employer.
