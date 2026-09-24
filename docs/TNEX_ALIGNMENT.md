# TNEX-aligned digital lending design

## Purpose and boundary

V7 demonstrates how the portfolio platform could support a mobile-first finance company with Cash Loan, BNPL and Business Loan journeys. Product limits and journey cues are based on public information accessed on 24 September 2026. Every record and performance result is synthetic. The implementation is not an official TNEX system, does not represent TNEX internal policy and is not affiliated with TNEX.

## Capability map

| Business need | V7 capability | Primary control |
|---|---|---|
| Fast digital onboarding | eKYC-to-offer orchestration | End-to-end p95 below five seconds |
| Multiple lending products | Versioned product configuration | Product-specific limit, PD and fraud thresholds |
| Thin-file decisions | Transaction, device, identity and sales features | Consent, freshness, owner and fallback for every feature |
| Fraud and credit risk | Two independent scores joined by policy | Fraud can decline or route to enhanced due diligence |
| Growth visibility | Six-stage acquisition funnel | Stage and start conversion reconciliation |
| Repeat borrowing | Dynamic limit actions | Increase, maintain, reduce or suppress with reason |
| Fair collections | Payment-ledger reconciliation | Automated reminders suppressed on unresolved payment |

## Public reference points

- TNEX app listing: https://play.google.com/store/apps/details?id=vn.tnex.consumer
- TNEX by MSB overview: https://www.msb.com.vn/khach-hang-ca-nhan/ngan-hang-so/tnex-by-msb-nen-tang-tai-chinh-so/
- TNEX personal-loan page: https://www.tnex.vn/san-pham/vay-tieu-dung-ca-nhan

Public sources can differ by product version or campaign. For that reason, limits are configuration, not model code, and require Product, Credit Risk, Legal and Compliance approval before production use.

## Production next steps

Replace synthetic distributions with approved data; complete privacy and consent assessment; validate fraud and credit models independently; calibrate cut-offs to risk appetite and unit economics; run shadow and canary releases; integrate the core ledger and payment-reconciliation service; and evidence customer-outcome monitoring before launch.

## V8 operating simulation

V8 adds an interactive decision contract and five operating views. The live response keeps product PD and fraud score separate, records the policy and model version, applies linked-identity signals, and returns reason codes. The optimizer reconciles interest income against credit loss, fraud loss, funding cost and operating cost. Its recommended strategy is constrained to an approval-rate range and a maximum fraud cut-off rather than selecting unconstrained profit alone. Collections actions prioritize payment reconciliation and customer vulnerability before automated contact.
