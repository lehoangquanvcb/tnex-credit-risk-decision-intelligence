# Production Readiness Checklist

| Control | Portfolio status | Production requirement |
|---|---|---|
| Representative customer data | Pending | Governed bank data and lineage approval |
| Independent validation | Demonstrated | Independent replication and signed report |
| Scorecard reconstruction | Pass | Required in every release gate |
| Policy versioning | Implemented | Credit authority approval and effective dates |
| Authentication and RBAC | Implemented | Enterprise identity provider integration |
| PostgreSQL audit trail | Configured | Managed database, encryption and retention policy |
| Override workflow | Implemented | Delegated authority and dual control |
| CI/CD gates | Implemented | Protected branches and approved runners |
| Load testing | Local gate | Production-like concurrency and endurance testing |
| Disaster recovery | Pending | RTO/RPO, failover and restore test |
| Security and privacy | Pending | Penetration test, secrets management and DPIA |
| Model Risk Committee | Pending | Formal approval before release |
# V5 release gates

- [x] Four-fold chronological walk-forward validation completed
- [x] Champion–challenger decision rule documented and executed
- [x] PD–LGD–EAD and profitability calculations reconciled
- [x] Base, Downturn and Severe stress scenarios completed
- [x] Candidate policy stored separately from active policy
- [ ] Replace synthetic data with governed bank development data
- [ ] Independent validation of model and strategy
- [ ] Credit Committee approval of cut-offs and pricing assumptions
- [ ] Controlled pilot with live monitoring and rollback criteria

## V6 additional evidence

- [x] Delayed-label monitoring separates mature and immature cohorts
- [x] LGD and EAD proxy models validated independently
- [x] Stage 1/2/3 and probability-weighted ECL calculated
- [x] Shadow and canary release gates documented
- [x] Incident register assigns severity, owner, action and SLA
- [x] Rollback drill reconciles model, policy, scores and service recovery
- [ ] Replace illustrative staging and macroeconomic assumptions with approved bank methodology
- [ ] Validate recovery cash flows, cure definitions and downturn LGD on real data
