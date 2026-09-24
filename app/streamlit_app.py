from pathlib import Path
import json, sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import FEATURES, assign_decision, score_from_pd
from src.v4_platform import pricing
from src.v8_digital_lending_os import live_decision

st.set_page_config(page_title="TNEX-aligned Digital Lending Decisioning", page_icon="💳", layout="wide")
st.title("TNEX-aligned Credit Risk & Decision Intelligence — V9")
st.caption("Product model pipelines • Portfolio risk appetite • Model governance • Customer outcomes • Risk committee reporting")
model = joblib.load(ROOT / "artifacts/retail_woe_scorecard_v4.joblib")
meta = json.loads((ROOT / "artifacts/model_metadata_v4.json").read_text())
payload=json.loads((ROOT/"reports/validation_payload.json").read_text())

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15,tab16,tab17,tab18,tab19=st.tabs(["Application scoring","Model validation","Strategy simulator","Profitability","Stress testing","Walk-forward","Production monitoring","IFRS 9 ECL","Deployment controls","Fairness","Behavioural EWS","Overrides","Governance","TNEX products","Real-time decisioning","Digital funnel","Lifecycle controls","TNEX V8","TNEX V9"])
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 20, 65, 35); income = st.number_input("Monthly income (VND)", 3_000_000, 500_000_000, 20_000_000, step=1_000_000)
        employment_months = st.number_input("Employment months", 0, 480, 60); employment_type = st.selectbox("Employment type", ["Salaried", "Self-employed", "Contract"])
    with c2:
        bureau_score = st.number_input("Bureau score", 300, 850, 650); existing_dti = st.slider("Existing DTI", 0.0, 0.95, .35, .01)
        inquiries_6m = st.number_input("Inquiries in 6 months", 0, 20, 1); home_ownership = st.selectbox("Home ownership", ["Owned", "Mortgage", "Rented", "Family"])
    with c3:
        requested_amount = st.number_input("Requested amount (VND)", 1_000_000, 2_000_000_000, 50_000_000, step=5_000_000)
        tenor_months = st.selectbox("Tenor", [6, 9, 12, 18, 24, 36], index=2); channel = st.selectbox("Channel", ["Mobile", "Branch", "Partner"])
    row = pd.DataFrame([[age,income,employment_months,bureau_score,existing_dti,requested_amount,tenor_months,inquiries_6m,employment_type,home_ownership,channel]], columns=FEATURES)
    pd12 = float(model.predict_proba(row)[0,1]); score = int(score_from_pd([pd12])[0]); decision = assign_decision(pd12)
    offer=pricing(pd12,income,requested_amount); decision=offer["decision"]
    a,b,c,d = st.columns(4); a.metric("12-month PD", f"{pd12:.2%}"); b.metric("Credit score", score); c.metric("Decision", decision); d.metric("Expected loss",f'VND {offer["expected_loss"]:,.0f}')
    st.write("**Point-based reason codes:** " + ", ".join(model.reason_codes(row)[0]))
    detail=pd.DataFrame(model.points_detail(row)[0]).sort_values("score_points").head(5)
    st.dataframe(detail[["feature","bin","woe","score_points"]],use_container_width=True,hide_index=True)
    st.info("Policy cut-offs: APPROVE < 8% PD • REVIEW 8%–18% • DECLINE ≥ 18%")
with tab2:
    cols = st.columns(5)
    vals = [("Status",meta["validation_status"]),("OOT AUC",f'{meta["auc"]:.3f}'),("Gini",f'{meta["gini"]:.3f}'),("KS",f'{meta["ks"]:.3f}'),("Brier",f'{meta["brier"]:.3f}')]
    for col,(k,v) in zip(cols,vals): col.metric(k,v)
    st.write("**Reconstruction control:**",payload["reconstruction"]["status"],"| Maximum PD gap:",f'{payload["reconstruction"]["max_model_vs_logit_pd_gap"]:.2e}')
    st.success("Validation covers discrimination, calibration proxy, stability and out-of-time performance. Full evidence is in reports/model_validation_pack.xlsx.")
with tab3:
    strategy=pd.DataFrame(payload["v5_strategy_grid"])
    c1,c2=st.columns(2)
    approve_max=c1.slider("Maximum PD for approval",.06,.13,.10,.01)
    valid_review=sorted(strategy.loc[strategy.approve_pd_max.round(2)==round(approve_max,2),"review_pd_max"].unique())
    review_max=c2.select_slider("Maximum PD for review",options=valid_review,value=min(valid_review,key=lambda x:abs(x-.18)))
    selected=strategy[(strategy.approve_pd_max.round(2)==round(approve_max,2))&(strategy.review_pd_max.round(2)==round(review_max,2))].iloc[0]
    k1,k2,k3,k4,k5=st.columns(5); k1.metric("Approval rate",f'{selected.approval_rate:.1%}'); k2.metric("Review rate",f'{selected.review_rate:.1%}'); k3.metric("Observed bad rate",f'{selected.observed_bad_rate:.1%}'); k4.metric("Expected profit",f'VND {selected.expected_profit/1e9:.1f}bn'); k5.metric("RAROC",f'{selected.portfolio_raroc:.1%}')
    curve=strategy[strategy.review_pd_max.round(2)==round(review_max,2)].set_index("approve_pd_max")
    st.line_chart(curve[["approval_rate","observed_bad_rate","portfolio_raroc"]])
    rec=payload["v5_summary"]; st.info(f'Recommended simulated policy: approve below {rec["recommended_approve_pd_max"]:.0%} PD and review below {rec["recommended_review_pd_max"]:.0%} PD, subject to committee approval and live-data validation.')
with tab4:
    pricing_view=pd.DataFrame(payload["v5_profitability"]); st.dataframe(pricing_view,use_container_width=True,hide_index=True)
    st.bar_chart(pricing_view.set_index("risk_band")[["expected_profit","expected_loss"]])
    st.caption("EAD is capped by requested amount and five times monthly income. Expected profit deducts expected loss, funding cost and operating cost; RAROC uses 10% economic capital.")
with tab5:
    stress=pd.DataFrame(payload["v5_stress_testing"]); st.dataframe(stress,use_container_width=True,hide_index=True)
    s1,s2,s3=st.columns(3)
    for col,scenario in zip([s1,s2,s3],["Base","Downturn","Severe"]):
        r=stress[stress.scenario==scenario].iloc[0]; col.metric(scenario,f'VND {r.expected_profit/1e9:.1f}bn',f'EL {r.expected_loss/1e9:.1f}bn')
    st.bar_chart(stress.set_index("scenario")[["expected_profit","expected_loss"]])
with tab6:
    wf=pd.DataFrame(payload["v5_champion_challenger"]); st.dataframe(wf,use_container_width=True,hide_index=True)
    st.line_chart(wf.pivot(index="window",columns="model",values="auc"))
    s=payload["v5_summary"]; a,b,c=st.columns(3); a.metric("Champion mean AUC",f'{s["champion_mean_auc"]:.3f}'); b.metric("Challenger mean AUC",f'{s["challenger_mean_auc"]:.3f}'); c.metric("Decision",s["challenger_decision"])
with tab7:
    mon=pd.DataFrame(payload["v6_delayed_monitoring"])
    m1,m2,m3,m4=st.columns(4);m1.metric("Applications",f'{mon.applications.sum():,.0f}');m2.metric("Mature months",f'{(mon.label_maturity_rate>=.8).sum()} / {len(mon)}');m3.metric("Latest PSI",f'{mon.iloc[-1].score_psi:.3f}');m4.metric("Latest status",mon.iloc[-1].status)
    st.line_chart(mon.set_index("month")[["mean_pd","observed_bad_rate","score_psi"]])
    st.dataframe(mon,use_container_width=True,hide_index=True)
    st.caption("AUC, KS and observed bad rate remain unavailable until the 12-month outcome window matures. Leading indicators are used in the interim.")
with tab8:
    stages=pd.DataFrame(payload["v6_ifrs9_stage_summary"]); scenarios=pd.DataFrame(payload["v6_macro_scenarios"])
    e1,e2,e3=st.columns(3);e1.metric("Weighted ECL",f'VND {stages.weighted_ecl.sum()/1e9:.2f}bn');e2.metric("Stage 2–3 share",f'{stages.loc[stages.stage!="Stage 1","accounts"].sum()/stages.accounts.sum():.1%}');e3.metric("Build status",payload["v6_summary"]["overall_status"])
    st.dataframe(stages,use_container_width=True,hide_index=True);st.bar_chart(stages.set_index("stage")[["total_ead","weighted_ecl"]])
    st.subheader("Macroeconomic scenarios");st.dataframe(scenarios,use_container_width=True,hide_index=True);st.bar_chart(scenarios.set_index("scenario")["total_ecl"])
    st.subheader("LGD and EAD validation");st.json({"LGD":payload["v6_lgd_validation"],"EAD":payload["v6_ead_validation"]})
with tab9:
    gates=pd.DataFrame(payload["v6_deployment_gates"]);incidents=pd.DataFrame(payload["v6_incidents"]);rollback=pd.DataFrame(payload["v6_rollback_drill"])
    d1,d2,d3=st.columns(3);d1.metric("Rollout status",payload["v6_summary"]["rollout_status"]);d2.metric("Rollback controls",f'{payload["v6_summary"]["rollback_controls_passed"]}/5 PASS');d3.metric("Open incidents",f'{(incidents.status=="OPEN").sum()}')
    st.subheader("Shadow and canary gates");st.dataframe(gates,use_container_width=True,hide_index=True)
    st.subheader("Incident register");st.dataframe(incidents,use_container_width=True,hide_index=True)
    st.subheader("Rollback evidence");st.dataframe(rollback,use_container_width=True,hide_index=True)
with tab10:
    fair=pd.DataFrame(payload["v41_fairness"]); dim=st.selectbox("Assessment dimension",fair.dimension.unique()); view=fair[fair.dimension==dim]
    st.dataframe(view,use_container_width=True,hide_index=True); st.bar_chart(view.set_index("group")[["approval_rate","bad_rate"]]); st.caption("A disparate impact ratio below 0.80 is flagged for review; it does not by itself establish unlawful discrimination.")
with tab11:
    ews=pd.DataFrame(payload["ews_summary"]); st.dataframe(ews,use_container_width=True,hide_index=True); st.bar_chart(ews.set_index("ews_status")["accounts"])
    accounts=pd.read_csv(ROOT/"data/behavioural_ews_sample.csv"); status=st.selectbox("EWS status",["RED","AMBER","GREEN"]); st.dataframe(accounts[accounts.ews_status==status].head(100),use_container_width=True,hide_index=True)
with tab12:
    queue=pd.read_csv(ROOT/"data/manual_review_queue.csv"); st.dataframe(queue[["application_id","pd_12m","decision","override_flag","override_direction","override_reason","final_decision","authority"]].head(200),use_container_width=True,hide_index=True)
    st.metric("Override rate",f'{queue.override_flag.mean():.1%}')
with tab13:
    gov=pd.DataFrame(payload["governance"]); st.dataframe(gov,use_container_width=True,hide_index=True); st.info("Committee approval and production release remain pending in this portfolio simulation.")
    st.subheader("Model registry")
    registry=json.loads((ROOT/"artifacts/model_registry.json").read_text()); st.json(registry)
    st.subheader("Active credit policy")
    st.json(payload["policy"])
with tab14:
    s=payload["v7_summary"]; a,b,c,d=st.columns(4); a.metric("Applications",f'{s["applications"]:,}'); b.metric("Products",s["products"]); c.metric("Approval rate",f'{s["approval_rate"]:.1%}'); d.metric("P95 decision",f'{s["p95_end_to_end_ms"]:.0f} ms')
    st.dataframe(pd.DataFrame(payload["v7_product_performance"]),use_container_width=True,hide_index=True)
    st.subheader("Versioned product policy"); st.dataframe(pd.DataFrame(payload["v7_product_config"]),use_container_width=True,hide_index=True)
    st.warning(s["disclaimer"])
with tab15:
    latency=pd.DataFrame(payload["v7_latency"]); st.bar_chart(latency.set_index("component")["p95_ms"]); st.dataframe(latency,use_container_width=True,hide_index=True)
    st.subheader("Alternative-data governance"); st.dataframe(pd.DataFrame(payload["v7_feature_governance"]),use_container_width=True,hide_index=True)
    st.caption("Consent, freshness and fallback are explicit controls. Missing alternative data never silently becomes an adverse decision.")
with tab16:
    funnel=pd.DataFrame(payload["v7_funnel"]); st.bar_chart(funnel.set_index("stage")["customers"]); st.dataframe(funnel,use_container_width=True,hide_index=True)
    st.metric("Start-to-disbursement conversion",f'{s["disbursement_conversion"]:.1%}')
with tab17:
    st.subheader("Repeat-customer limit actions"); st.dataframe(pd.DataFrame(payload["v7_lifecycle"]),use_container_width=True,hide_index=True)
    st.subheader("Payment and collections controls"); st.dataframe(pd.DataFrame(payload["v7_collections"]),use_container_width=True,hide_index=True)
    st.info("Unresolved payment-posting exceptions suppress automated reminders until the ledger is reconciled.")
with tab18:
    section=st.radio("V8 module",["Live decision","Risk–profit optimizer","Fraud network","Collections NBA","Command center"],horizontal=True)
    if section=="Live decision":
        c1,c2,c3=st.columns(3)
        product=c1.selectbox("Product",["CASH_LOAN","BNPL","BUSINESS_LOAN"]); income=c1.number_input("Monthly income (VND)",3_000_000,500_000_000,20_000_000,step=1_000_000); requested=c1.number_input("Requested amount (VND)",1_000_000,500_000_000,30_000_000,step=1_000_000)
        bureau=c2.number_input("Bureau score",300,850,650); dti=c2.slider("Existing DTI",0.0,.95,.35,.01); identity=c2.slider("Identity match",0.0,1.0,.94,.01)
        velocity=c3.number_input("Applications in 24 hours",0,20,0); shared=c3.number_input("Linked device accounts",0,20,0); txn=c3.number_input("Transactions in 90 days",0,500,52)
        if st.button("Run digital decision",type="primary"):
            app={"product":product,"age":35,"monthly_income":income,"employment_months":60,"bureau_score":bureau,"existing_dti":dti,"requested_amount":requested,"tenor_months":12,"inquiries_6m":1,"employment_type":"Salaried","home_ownership":"Rented","channel":"Mobile","identity_match_score":identity,"device_age_days":180,"application_velocity_24h":velocity,"failed_txn_rate":.01,"transaction_count_90d":txn,"income_regularly":1,"prior_on_time_ratio":.95,"sales_monthly":income*4 if product=="BUSINESS_LOAN" else 0,"shared_device_accounts":shared,"shared_payout_accounts":0}
            result=live_decision(app,model); a,b,c,d=st.columns(4);a.metric("Decision",result["decision"]);b.metric("Product PD",f'{result["product_pd"]:.2%}');c.metric("Fraud score",f'{result["fraud_score"]:.2%}');d.metric("Approved limit",f'VND {result["approved_limit_vnd"]:,.0f}');st.write("**Reason codes:** "+"; ".join(result["reason_codes"]));st.json(result)
    elif section=="Risk–profit optimizer":
        opt=pd.DataFrame(payload["v8_optimizer"]);rec=payload["v8_recommended_strategy"];a,b,c,d=st.columns(4);a.metric("Credit cut-off",f'{rec["credit_pd_cutoff"]:.0%}');b.metric("Fraud cut-off",f'{rec["fraud_cutoff"]:.0%}');c.metric("Approval",f'{rec["approval_rate"]:.1%}');d.metric("RAROC",f'{rec["raroc"]:.1%}');st.dataframe(opt,use_container_width=True,hide_index=True)
    elif section=="Fraud network": st.dataframe(pd.DataFrame(payload["v8_fraud_network"]),use_container_width=True,hide_index=True)
    elif section=="Collections NBA": st.dataframe(pd.DataFrame(payload["v8_collections_nba"]),use_container_width=True,hide_index=True);st.success("Payment-pending accounts have contact suppression enabled.")
    else:
        mon=pd.DataFrame(payload["v8_monitoring"]);st.dataframe(mon,use_container_width=True,hide_index=True);st.subheader("Product model calibration");st.dataframe(pd.DataFrame(payload["v8_product_models"]),use_container_width=True,hide_index=True);st.warning(payload["v8_summary"]["disclaimer"])
with tab19:
    view=st.radio("V9 executive view",["Risk committee","Product models","Risk appetite","Model governance","Customer outcomes"],horizontal=True)
    s9=payload["v9_summary"]
    if view=="Risk committee":
        a,b,c,d=st.columns(4);a.metric("Models passing",f'{s9["models_passing"]}/3');b.metric("Portfolio capacity",f'VND {s9["portfolio_capacity_vnd"]/1e9:,.0f}bn');c.metric("Portfolio RAROC",f'{s9["recommended_raroc"]:.1%}');d.metric("Open findings",s9["open_findings"]);st.dataframe(pd.DataFrame(payload["v9_committee_pack"]),use_container_width=True,hide_index=True);st.warning(s9["disclaimer"])
    elif view=="Product models": st.dataframe(pd.DataFrame(payload["v9_product_models"]),use_container_width=True,hide_index=True);st.info("Business Loan remains in REVIEW because its synthetic OOT sample and discriminatory power are not yet sufficient for production approval.")
    elif view=="Risk appetite":
        r=payload["v9_recommended_allocation"];a,b,c=st.columns(3);a.metric("Cash Loan",f'{r["cash_loan_share"]:.0%}');b.metric("BNPL",f'{r["bnpl_share"]:.0%}');c.metric("Business Loan",f'{r["business_loan_share"]:.0%}');st.dataframe(pd.DataFrame(payload["v9_risk_appetite_grid"]),use_container_width=True,hide_index=True)
    elif view=="Model governance": st.subheader("Model inventory");st.dataframe(pd.DataFrame(payload["v9_model_inventory"]),use_container_width=True,hide_index=True);st.subheader("Validation findings");st.dataframe(pd.DataFrame(payload["v9_validation_findings"]),use_container_width=True,hide_index=True)
    else: st.subheader("Responsible-lending controls");st.dataframe(pd.DataFrame(payload["v9_customer_controls"]),use_container_width=True,hide_index=True);st.subheader("Segment outcomes");st.dataframe(pd.DataFrame(payload["v9_customer_outcomes"]),use_container_width=True,hide_index=True)
