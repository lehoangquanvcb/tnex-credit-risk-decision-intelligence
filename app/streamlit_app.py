from pathlib import Path
from datetime import date, datetime, timezone
import json, sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import FEATURES, assign_decision, score_from_pd
from src.v4_platform import pricing

st.set_page_config(page_title="TNEX Retail Credit Risk Workbench", page_icon="💳", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --tnex-blue:#0b2239; --tnex-panel:#102c46; --tnex-border:#294c68; --tnex-cyan:#19b5c5; --tnex-green:#22c55e; --tnex-amber:#f59e0b; --tnex-red:#ef4444; }
.stApp { background:linear-gradient(180deg,#071a2b 0%,#0a2136 100%); color:#f8fafc; }
[data-testid="stSidebar"] { background:#071c2f; border-right:1px solid #294c68; }
[data-testid="stSidebar"] .block-container { padding-top:1.2rem; }
.block-container { max-width:1800px; padding-top:2.4rem; padding-bottom:2rem; }
h1,h2,h3 { letter-spacing:-.02em; color:#f8fafc !important; }
[data-testid="stMetric"] { background:#102c46; border:1px solid #294c68; border-radius:12px; padding:15px 17px; min-height:112px; box-shadow:0 5px 18px rgba(0,0,0,.14); }
[data-testid="stMetricLabel"] { color:#b7c8d8; font-weight:700; }
[data-testid="stMetricValue"] { color:#fff; font-weight:800; }
[data-testid="stMetricDelta"] { font-weight:700; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #294c68; border-radius:10px; overflow:hidden; }
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] { border-color:#294c68 !important; border-radius:12px !important; }
.stAlert { border-radius:10px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#102c46; border-color:#294c68; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.42rem .55rem; margin:.09rem 0; border-radius:8px; border:1px solid transparent; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#123653; border-color:#294c68; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:linear-gradient(90deg,#0969da,#0ea5a8); border-color:#38bdf8; font-weight:800; }
.tnex-brand { text-align:center; padding:8px 4px 18px; border-bottom:1px solid #294c68; margin-bottom:12px; }
.tnex-logo { width:48px; height:48px; border-radius:14px; display:inline-flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#00b5ad,#2563eb); font-size:25px; box-shadow:0 8px 20px rgba(0,181,173,.25); }
.tnex-brand h2 { font-size:18px; margin:9px 0 1px; }
.tnex-brand p { color:#8eacc4; font-size:11px; margin:0; }
.hero { background:linear-gradient(115deg,#103b5a 0%,#0d3552 55%,#075d67 100%); border:1px solid #34708b; border-radius:15px; padding:24px 22px 20px; margin:8px 0 14px; box-shadow:0 9px 28px rgba(0,0,0,.18); overflow:visible; }
.hero-kicker { color:#65dce5; font-size:11px; line-height:1.5; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:5px; }
.hero h1 { margin:3px 0 3px; font-size:29px; line-height:1.25; }
.hero p { margin:0; color:#c3d6e5; font-size:13px; }
.section-title { font-size:16px; font-weight:800; margin:15px 0 8px; color:#f8fafc; }
.kpi-card { background:linear-gradient(145deg,#102f4b,#10283f); border:1px solid #315572; border-radius:12px; padding:14px 14px 12px; min-height:118px; box-shadow:0 5px 18px rgba(0,0,0,.15); }
.kpi-icon { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#164e72; font-size:16px; margin-bottom:8px; }
.kpi-label { color:#a9bfd0; font-size:11px; font-weight:700; min-height:27px; }
.kpi-value { color:#fff; font-size:23px; line-height:1.1; font-weight:850; margin:4px 0; }
.kpi-delta { font-size:10px; font-weight:700; color:#50df8a; }
.kpi-delta.warn { color:#fbbf24; } .kpi-delta.bad { color:#ff6b6b; }
.panel-title { color:#f8fafc; font-size:14px; font-weight:800; margin-bottom:7px; }
.allocation-row { display:grid; grid-template-columns:105px 1fr 48px; gap:10px; align-items:center; margin:17px 0; font-size:12px; font-weight:700; }
.allocation-track { background:#071a2b; height:15px; border-radius:999px; overflow:hidden; border:1px solid #294c68; }
.allocation-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb,#22c1c3); }
.status-pill { display:inline-block; border-radius:999px; padding:3px 9px; font-size:10px; font-weight:800; background:#164e3a; color:#75f0a5; }
@media (max-width:900px) { .hero h1{font-size:22px}.kpi-value{font-size:19px}.block-container{padding-left:.8rem;padding-right:.8rem} }
</style>
""", unsafe_allow_html=True)
model = joblib.load(ROOT / "artifacts/retail_woe_scorecard_v4.joblib")
meta = json.loads((ROOT / "artifacts/model_metadata_v4.json").read_text())
payload=json.loads((ROOT/"reports/validation_payload.json").read_text())

@st.cache_data
def load_csv(name):
    return pd.read_csv(ROOT / "data" / name)

NAVIGATION = [
    "🏠  Executive overview",
    "▤  Portfolio overview",
    "◴  Vintage & delinquency",
    "◉  Collections & EWS",
    "◎  Application scoring",
    "◫  Model validation",
    "⌁  Strategy & profitability",
    "Σ  IFRS 9 ECL",
    "⚡  Production & decisioning",
    "▣  Products & funnel",
    "◌  MIS & risk indicators",
    "◇  Governance & controls",
    "▦  Data governance"
]
st.sidebar.markdown('<div class="tnex-brand"><div class="tnex-logo">💳</div><h2>TNEX CREDIT RISK</h2><p>Retail Decision Intelligence Workbench</p></div>', unsafe_allow_html=True)
st.sidebar.caption("NAVIGATION")
selected_label = st.sidebar.radio("Navigation",NAVIGATION,key="main_navigation",label_visibility="collapsed")
selected_module=selected_label.split("  ",1)[-1]
selected_group=selected_module.title()
st.sidebar.markdown("---")
st.sidebar.caption("V15 • Production demo & control layer")

PAGE_SUBTITLE = {
    "Executive overview":"Portfolio risk, product economics, model health and management actions",
    "Portfolio overview":"Portfolio growth, approval quality, exposure and risk segmentation",
    "Vintage & delinquency":"Cohort performance, DPD stock, migration and behavioural deterioration",
    "Collections & EWS":"Next-best-action, behavioural warning signals and lifecycle safeguards",
    "MIS & risk indicators":"Management information, portfolio KRIs and operational risk signals",
    "Application scoring":"Applicant-level PD, score, affordability and explainable decisioning",
    "Model validation":"Independent validation, walk-forward evidence and customer fairness",
    "Strategy & profitability":"Optimize approval, expected loss, RAROC and stress resilience",
    "IFRS 9 ECL":"Staging, scenario-weighted ECL and LGD/EAD validation",
    "Production & decisioning":"Production monitoring, deployment gates and real-time controls",
    "Products & funnel":"Product policy, performance, conversion and customer journey",
    "Governance & controls":"Model governance, overrides, findings and accountable actions",
    "Data governance":"Data quality controls and common business definitions"
}
st.markdown(f'<div class="hero"><div class="hero-kicker">TNEX • CONSUMER CREDIT RISK • V15</div><h1>{selected_group}</h1><p>{PAGE_SUBTITLE[selected_module]}</p></div>', unsafe_allow_html=True)

def kpi_card(column, icon, label, value, note, tone="good"):
    column.markdown(f'<div class="kpi-card"><div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-delta {tone}">{note}</div></div>', unsafe_allow_html=True)

if selected_module == "Executive overview":
    s9=payload["v9_summary"]
    decisions=load_csv("tnex_decisions_v7.csv")
    decisions["application_date"]=pd.to_datetime(decisions.application_date)
    decisions["risk_band"]=pd.cut(decisions.decision_pd,[-1,.05,.10,.18,1],labels=["Low","Medium","High","Very High"])
    min_date=decisions.application_date.min().date(); max_date=decisions.application_date.max().date()
    if "overview_date" in st.session_state and not min_date<=st.session_state.overview_date<=max_date:
        st.session_state.overview_date=max_date
    def reset_overview_filters():
        st.session_state.overview_date=max_date
        st.session_state.overview_product="All"; st.session_state.overview_channel="All"
        st.session_state.overview_decision="All"; st.session_state.overview_risk="All"
    with st.container(border=True):
        f1,f2,f3,f4,f5,f6=st.columns([1.05,1.15,1,1,1,0.55])
        as_of=f1.date_input("As of date",max_date,min_value=min_date,max_value=max_date,key="overview_date")
        product_filter=f2.selectbox("Product",["All"]+sorted(decisions["product"].unique().tolist()),key="overview_product")
        channel_filter=f3.selectbox("Channel",["All"]+sorted(decisions.channel.unique().tolist()),key="overview_channel")
        decision_filter=f4.selectbox("Decision",["All"]+sorted(decisions.decision.unique().tolist()),key="overview_decision")
        risk_filter=f5.selectbox("Risk band",["All","Low","Medium","High","Very High"],key="overview_risk")
        f6.markdown("<br>",unsafe_allow_html=True); f6.button("↻ Clear",use_container_width=True,key="overview_clear",on_click=reset_overview_filters)
    filtered=decisions[decisions.application_date.dt.date<=as_of].copy()
    for column,value in [("product",product_filter),("channel",channel_filter),("decision",decision_filter),("risk_band",risk_filter)]:
        if value!="All": filtered=filtered[filtered[column].astype(str)==value]
    if filtered.empty:
        st.error("No applications match the selected filters. Clear or broaden the selection.")
        st.stop()
    approved=filtered[filtered.decision=="APPROVE"]
    exposure=filtered.requested_amount.sum()
    expected_loss=(approved.decision_pd*.65*approved.approved_limit_vnd).sum()
    cards=st.columns(6)
    kpi_card(cards[0],"▤","Applications",f'{len(filtered):,}',"Filtered population")
    kpi_card(cards[1],"₫","Requested exposure",f'{exposure/1e9:,.1f}B',"VND application volume")
    kpi_card(cards[2],"✓","Approval rate",f'{(filtered.decision=="APPROVE").mean():.1%}',"Filtered policy outcome")
    kpi_card(cards[3],"◉","Mean decision PD",f'{filtered.decision_pd.mean():.1%}',"Post-product adjustment","warn")
    kpi_card(cards[4],"△","Expected credit loss",f'{expected_loss/1e9:,.2f}B',"Approved exposure","warn")
    kpi_card(cards[5],"⚡","P95 decision latency",f'{filtered.end_to_end_ms.quantile(.95):.0f} ms',"End-to-end SLA")
    st.markdown('<div class="section-title">Portfolio intelligence</div>',unsafe_allow_html=True)
    left,mid,right=st.columns([1.15,1.15,1])
    with left:
        st.markdown('<div class="panel-title">Filtered product allocation</div>',unsafe_allow_html=True)
        shares=filtered.groupby("product").requested_amount.sum().div(max(exposure,1))
        allocation=[(name.replace("_"," ").title(),share) for name,share in shares.items()]
        allocation_html="".join([f'<div class="allocation-row"><span>{name}</span><div class="allocation-track"><div class="allocation-fill" style="width:{share:.0%}"></div></div><b>{share:.0%}</b></div>' for name,share in allocation])
        st.markdown(f'<div style="background:#10283f;border:1px solid #294c68;border-radius:10px;padding:18px 16px;height:270px">{allocation_html}<div style="color:#8eacc4;font-size:11px;margin-top:18px">Share of requested exposure after applying all dashboard filters.</div></div>',unsafe_allow_html=True)
    with mid:
        st.markdown('<div class="panel-title">Product model health</div>',unsafe_allow_html=True)
        models=pd.DataFrame(payload["v9_product_models"])
        if product_filter!="All": models=models[models["product"]==product_filter]
        health=models[["product","oot_auc","oot_gini","oot_ks","calibration_gap","status","next_review"]].copy()
        health.columns=["Product","AUC","Gini","KS","Cal. gap","Status","Next review"]
        st.dataframe(health,use_container_width=True,hide_index=True,height=270)
    with right:
        st.markdown('<div class="panel-title">Key risk indicators</div>',unsafe_allow_html=True)
        indicators=pd.DataFrame([
            ["Observed bad rate",f'{filtered.default_12m.mean():.2%}',"PASS" if filtered.default_12m.mean()<.15 else "WATCH"],
            ["High-risk share",f'{filtered.risk_band.astype(str).isin(["High","Very High"]).mean():.2%}',"MONITOR"],
            ["Fraud decline rate",f'{((filtered.fraud_score>.30)&(filtered.decision=="DECLINE")).mean():.2%}',"MONITOR"],
            ["Stage 2–3 share",f'{payload["v6_summary"]["stage_2_3_share"]:.2%}',"PASS"],
            ["Open findings",str(s9["open_findings"]),"ACTION"]
        ],columns=["Metric","Value","Status"])
        st.dataframe(indicators,use_container_width=True,hide_index=True,height=270)
    st.markdown('<div class="section-title">Performance, customer journey and controls</div>',unsafe_allow_html=True)
    p1,p2,p3=st.columns([1.1,1.15,1])
    with p1:
        st.markdown('<div class="panel-title">Decision outcomes</div>',unsafe_allow_html=True)
        st.bar_chart(filtered.decision.value_counts(),height=255,color="#22c1c3")
    with p2:
        st.markdown('<div class="panel-title">Risk-band distribution</div>',unsafe_allow_html=True)
        st.bar_chart(filtered.risk_band.value_counts(sort=False),height=255,color="#f59e0b")
    with p3:
        st.markdown('<div class="panel-title">Product performance</div>',unsafe_allow_html=True)
        performance=filtered.groupby("product").agg(applications=("application_id","count"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),mean_decision_pd=("decision_pd","mean"),p95_latency_ms=("end_to_end_ms",lambda x:x.quantile(.95))).reset_index()
        st.dataframe(performance,use_container_width=True,hide_index=True,height=255)
    st.markdown('<div class="section-title">Executive decisions and actions</div>',unsafe_allow_html=True)
    a1,a2=st.columns([1.35,1])
    with a1:
        st.dataframe(pd.DataFrame(payload["v9_committee_pack"]),use_container_width=True,hide_index=True)
    with a2:
        st.warning("**Business Loan model:** retain REVIEW status until OOT sample and discriminatory power meet production thresholds.")
        st.info("**Risk appetite:** deploy the recommended product allocation subject to committee approval and live-data validation.")
        st.success("**Deployment:** rollback controls passed; maintain staged rollout and weekly drift monitoring.")

if selected_module == "Portfolio overview":
    prod=load_csv("production_monitoring_sample.csv")
    with st.container(border=True):
        x1,x2,x3,x4=st.columns(4)
        channel_pick=x1.selectbox("Channel",["All"]+sorted(prod.channel.unique().tolist()),key="portfolio_channel")
        decision_pick=x2.selectbox("Decision",["All"]+sorted(prod.decision.unique().tolist()),key="portfolio_decision")
        employment_pick=x3.selectbox("Employment",["All"]+sorted(prod.employment_type.unique().tolist()),key="portfolio_employment")
        month_pick=x4.selectbox("Reporting month",["All"]+sorted(prod.month.astype(str).unique().tolist()),key="portfolio_month")
    view=prod.copy()
    for col,val in [("channel",channel_pick),("decision",decision_pick),("employment_type",employment_pick),("month",month_pick)]:
        if val!="All": view=view[view[col].astype(str)==str(val)]
    cards=st.columns(6)
    kpi_card(cards[0],"▤","Applications",f'{len(view):,}',"Filtered portfolio")
    kpi_card(cards[1],"₫","Requested exposure",f'{view.requested_amount.sum()/1e9:,.1f}B',"VND application volume")
    kpi_card(cards[2],"✓","Approval rate",f'{(view.decision=="APPROVE").mean():.1%}',"Policy outcome")
    kpi_card(cards[3],"◉","Observed bad rate",f'{view.default_12m.mean():.1%}',"12-month outcome","warn")
    kpi_card(cards[4],"◎","Mean predicted PD",f'{view.pd_12m.mean():.1%}',"Model expectation")
    kpi_card(cards[5],"◆","Average score",f'{view.credit_score.mean():.0f}',"Portfolio quality")
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="section-title">Monthly applications and exposure</div>',unsafe_allow_html=True)
        monthly=view.groupby("month").agg(applications=("application_id","count"),exposure_bn=("requested_amount",lambda x:x.sum()/1e9))
        st.line_chart(monthly,height=300)
    with b:
        st.markdown('<div class="section-title">Decision and channel mix</div>',unsafe_allow_html=True)
        mix=pd.crosstab(view.channel,view.decision)
        st.bar_chart(mix,height=300)
    st.markdown('<div class="section-title">Portfolio risk segmentation</div>',unsafe_allow_html=True)
    view=view.assign(risk_band=pd.cut(view.pd_12m,[-1,.05,.10,.18,1],labels=["Low","Medium","High","Very High"]))
    risk=view.groupby("risk_band",observed=False).agg(applications=("application_id","count"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),bad_rate=("default_12m","mean"),mean_pd=("pd_12m","mean"),exposure_vnd=("requested_amount","sum")).reset_index()
    st.dataframe(risk,use_container_width=True,hide_index=True)

if selected_module == "Vintage & delinquency":
    st.markdown('<div class="section-title">Vintage performance</div>',unsafe_allow_html=True)
    q=load_csv("manual_review_queue.csv")
    vintage=q.groupby("month").agg(bookings=("application_id","count"),approval_rate=("final_decision",lambda x:(x=="APPROVE").mean()),fpd30_rate=("fpd30","mean"),mob3_30plus_rate=("mob3_30plus","mean"),bad_rate_12m=("default_12m","mean"),mean_pd=("pd_12m","mean")).reset_index()
    cards=st.columns(5)
    kpi_card(cards[0],"▥","Cohorts",f'{len(vintage)}',"Monthly vintages")
    kpi_card(cards[1],"⚠","Latest FPD30",f'{vintage.iloc[-1].fpd30_rate:.1%}',"Early payment risk","warn")
    kpi_card(cards[2],"↗","Latest MOB3 30+",f'{vintage.iloc[-1].mob3_30plus_rate:.1%}',"Seasoning indicator","warn")
    kpi_card(cards[3],"◉","12M bad rate",f'{q.default_12m.mean():.1%}',"Observed portfolio")
    kpi_card(cards[4],"✓","Approval rate",f'{(q.final_decision=="APPROVE").mean():.1%}',"Final decisions")
    c1,c2=st.columns([1.35,1])
    with c1:
        st.markdown('<div class="section-title">Vintage delinquency curves</div>',unsafe_allow_html=True)
        st.line_chart(vintage.set_index("month")[["fpd30_rate","mob3_30plus_rate","bad_rate_12m"]],height=340)
    with c2:
        st.markdown('<div class="section-title">Cohort quality table</div>',unsafe_allow_html=True)
        st.dataframe(vintage,use_container_width=True,hide_index=True,height=340)
    st.caption("Portfolio simulation based on the available labelled review sample; replace with production booking cohorts when TNEX monthly snapshots become available.")

if selected_module == "Vintage & delinquency":
    st.markdown('<div class="section-title">DPD stock and roll-rate control</div>',unsafe_allow_html=True)
    ews=load_csv("behavioural_ews_sample.csv")
    labels=["Current","1–30","31–60","61–90","90+"]
    ews["dpd_bucket"]=pd.cut(ews.days_past_due,[-1,0,30,60,90,10_000],labels=labels)
    dist=ews.groupby("dpd_bucket",observed=False).agg(accounts=("account_id","count"),mean_pd=("behavioural_pd","mean"),default_rate=("default_90d","mean"),mean_utilisation=("utilisation","mean")).reset_index()
    cards=st.columns(5)
    for col,bucket in zip(cards,labels):
        row=dist[dist.dpd_bucket==bucket].iloc[0]; kpi_card(col,"◌",bucket,f'{int(row.accounts):,}',f'Default {row.default_rate:.1%}',"bad" if bucket in ["61–90","90+"] else "warn")
    a,b=st.columns([1,1.35])
    with a:
        st.markdown('<div class="section-title">Current DPD stock</div>',unsafe_allow_html=True); st.bar_chart(dist.set_index("dpd_bucket")["accounts"],height=330)
    with b:
        st.markdown('<div class="section-title">Illustrative one-month migration matrix</div>',unsafe_allow_html=True)
        matrix=pd.DataFrame([[.955,.024,.013,.006,.002],[.361,.312,.283,.044,0],[.171,.247,.308,.275,0],[.091,.109,.218,.473,.109],[.022,.018,.035,.115,.810]],index=labels,columns=labels)
        st.dataframe(matrix.style.format("{:.1%}").background_gradient(cmap="RdYlGn_r",axis=None),use_container_width=True,height=330)
    st.warning("Migration matrix is an illustrative control benchmark because the current package contains one behavioural snapshot. Actual roll and cure rates require account-level month-on-month DPD histories.")

if selected_module == "Vintage & delinquency":
    st.markdown('<div class="section-title">Behavioural delinquency analysis</div>',unsafe_allow_html=True)
    ews=load_csv("behavioural_ews_sample.csv")
    ews["dpd_bucket"]=pd.cut(ews.days_past_due,[-1,0,30,60,90,10_000],labels=["Current","1–30","31–60","61–90","90+"])
    summary=ews.groupby(["ews_status","dpd_bucket"],observed=False).agg(accounts=("account_id","count"),mean_pd=("behavioural_pd","mean"),default_rate=("default_90d","mean"),utilisation=("utilisation","mean"),payment_ratio=("payment_ratio","mean")).reset_index()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("30+ DPD",f'{(ews.days_past_due>30).mean():.1%}'); c2.metric("90+ DPD",f'{(ews.days_past_due>90).mean():.1%}'); c3.metric("Red EWS",f'{(ews.ews_status=="RED").mean():.1%}'); c4.metric("Mean behavioural PD",f'{ews.behavioural_pd.mean():.1%}')
    a,b=st.columns(2)
    with a: st.markdown('<div class="section-title">Accounts by DPD and EWS</div>',unsafe_allow_html=True); st.bar_chart(summary.pivot(index="dpd_bucket",columns="ews_status",values="accounts"),height=330)
    with b: st.markdown('<div class="section-title">Risk severity by DPD bucket</div>',unsafe_allow_html=True); st.line_chart(ews.groupby("dpd_bucket",observed=False)[["behavioural_pd","default_90d","utilisation"]].mean(),height=330)
    st.dataframe(summary,use_container_width=True,hide_index=True)

if selected_module == "Collections & EWS":
    st.markdown('<div class="section-title">Collection next-best-action</div>',unsafe_allow_html=True)
    nba=load_csv("collections_nba_v8.csv"); controls=load_csv("collections_controls_v7.csv")
    cards=st.columns(5)
    kpi_card(cards[0],"◉","Collection accounts",f'{len(nba):,}',"NBA population")
    kpi_card(cards[1],"⚠","Payment pending",f'{nba.payment_pending.mean():.1%}',"Contact suppression","warn")
    kpi_card(cards[2],"◇","Vulnerable customers",f'{nba.vulnerability_flag.mean():.1%}',"Treatment safeguards","warn")
    kpi_card(cards[3],"✓","Contact allowed",f'{nba.contact_allowed.mean():.1%}',"Controlled outreach")
    kpi_card(cards[4],"↗","Mean DPD",f'{nba.dpd.mean():.1f}',"Collection severity")
    a,b=st.columns([1.15,1])
    with a: st.markdown('<div class="section-title">Next-best-action allocation</div>',unsafe_allow_html=True); st.bar_chart(nba.nba.value_counts(),height=320)
    with b: st.markdown('<div class="section-title">Collection controls</div>',unsafe_allow_html=True); st.dataframe(controls,use_container_width=True,hide_index=True,height=320)
    st.markdown('<div class="section-title">Prioritized work queue</div>',unsafe_allow_html=True)
    st.dataframe(nba.sort_values(["vulnerability_flag","dpd"],ascending=[False,False]).head(150),use_container_width=True,hide_index=True)

if selected_module == "MIS & risk indicators":
    st.markdown('<div class="section-title">Key risk indicators</div>',unsafe_allow_html=True)
    mon=load_csv("delayed_monitoring_v6.csv"); latest=mon.iloc[-1]; incidents=load_csv("model_incidents_v6.csv"); ews=load_csv("behavioural_ews_sample.csv")
    cards=st.columns(6)
    kpi_card(cards[0],"◎","Score PSI",f'{latest.score_psi:.3f}',str(latest.status),"warn")
    kpi_card(cards[1],"◉","Mean PD",f'{latest.mean_pd:.1%}',"Latest production month")
    kpi_card(cards[2],"⚠","30+ DPD",f'{(ews.days_past_due>30).mean():.1%}',"Behavioural snapshot","warn")
    kpi_card(cards[3],"◆","Red EWS",f'{(ews.ews_status=="RED").mean():.1%}',"Immediate review","bad")
    kpi_card(cards[4],"△","Open incidents",f'{(incidents.status=="OPEN").sum()}',"Operational controls","warn")
    kpi_card(cards[5],"✓","Rollback tests",f'{payload["v6_summary"]["rollback_controls_passed"]}/5',"Deployment resilience")
    st.markdown('<div class="section-title">Risk signal trend</div>',unsafe_allow_html=True); st.line_chart(mon.set_index("month")[["mean_pd","observed_bad_rate","score_psi","calibration_gap_when_mature"]],height=340)
    st.markdown('<div class="section-title">Incident and action register</div>',unsafe_allow_html=True); st.dataframe(incidents,use_container_width=True,hide_index=True)

if selected_module == "MIS & risk indicators":
    st.markdown('<div class="section-title">Management information dashboard</div>',unsafe_allow_html=True)
    prod=load_csv("production_monitoring_sample.csv"); monthly=prod.groupby("month").agg(applications=("application_id","count"),exposure_vnd=("requested_amount","sum"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),mean_pd=("pd_12m","mean"),bad_rate=("default_12m","mean"),mean_score=("credit_score","mean")).reset_index()
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Applications",f'{len(prod):,}');c2.metric("Exposure",f'VND {prod.requested_amount.sum()/1e9:,.1f}B');c3.metric("Approval",f'{(prod.decision=="APPROVE").mean():.1%}');c4.metric("Bad rate",f'{prod.default_12m.mean():.1%}');c5.metric("Mean score",f'{prod.credit_score.mean():.0f}')
    a,b=st.columns(2)
    with a: st.markdown('<div class="section-title">Volumes and exposure</div>',unsafe_allow_html=True); st.line_chart(monthly.set_index("month")[["applications","exposure_vnd"]],height=320)
    with b: st.markdown('<div class="section-title">Approval and credit quality</div>',unsafe_allow_html=True); st.line_chart(monthly.set_index("month")[["approval_rate","mean_pd","bad_rate"]],height=320)
    st.dataframe(monthly,use_container_width=True,hide_index=True)
    st.download_button("Download MIS extract",monthly.to_csv(index=False).encode("utf-8"),"tnex_mis_monthly.csv","text/csv",key="download_mis")

if selected_module == "Application scoring":
    st.markdown('<div class="section-title">Applicant, product and policy inputs</div>',unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        product=st.selectbox("TNEX product",["CASH_LOAN","BNPL","BUSINESS_LOAN"],key="score_product")
        age = st.number_input("Age", 20, 65, 35,key="score_age"); income = st.number_input("Monthly income (VND)", 3_000_000, 500_000_000, 20_000_000, step=1_000_000,key="score_income")
        employment_months = st.number_input("Employment months", 0, 480, 60,key="score_employment_months"); employment_type = st.selectbox("Employment type", ["Salaried", "Self-employed", "Contract"],key="score_employment_type")
    with c2:
        bureau_score = st.number_input("Bureau score", 300, 850, 650,key="score_bureau"); existing_dti = st.slider("Existing DTI", 0.0, 0.95, .35, .01,key="score_dti")
        inquiries_6m = st.number_input("Inquiries in 6 months", 0, 20, 1,key="score_inquiries"); home_ownership = st.selectbox("Home ownership", ["Owned", "Mortgage", "Rented", "Family"],key="score_home")
        identity_match=st.slider("Identity match score",0.0,1.0,.94,.01,key="score_identity")
    with c3:
        requested_amount = st.number_input("Requested amount (VND)", 1_000_000, 2_000_000_000, 50_000_000, step=5_000_000,key="score_amount")
        tenor_months = st.selectbox("Tenor", [6, 9, 12, 18, 24, 36], index=2,key="score_tenor"); channel = st.selectbox("Channel", ["Mobile", "Branch", "Partner"],key="score_channel")
        velocity=st.number_input("Applications in 24 hours",0,20,0,key="score_velocity")
    row = pd.DataFrame([[age,income,employment_months,bureau_score,existing_dti,requested_amount,tenor_months,inquiries_6m,employment_type,home_ownership,channel]], columns=FEATURES)
    pd12 = float(model.predict_proba(row)[0,1]); score = int(score_from_pd([pd12])[0])
    config=pd.DataFrame(payload["v7_product_config"]).set_index("product").loc[product]
    offer=pricing(pd12,income,requested_amount)
    fraud_score=min(1.0,max(0.0,.55*(1-identity_match)+.035*velocity+.04*max(inquiries_6m-2,0)+(.08 if channel=="Partner" else 0)))
    policy_reasons=[]
    if fraud_score>config.fraud_max: policy_reasons.append("FRAUD_THRESHOLD_EXCEEDED")
    if existing_dti>.65: policy_reasons.append("AFFORDABILITY_DTI_EXCEEDED")
    if income<5_000_000: policy_reasons.append("MINIMUM_INCOME_NOT_MET")
    if pd12>=config.review_pd_max: policy_reasons.append("PD_ABOVE_REVIEW_CUTOFF")
    if "FRAUD_THRESHOLD_EXCEEDED" in policy_reasons or pd12>=config.review_pd_max: decision="DECLINE"
    elif pd12<config.approve_pd_max and existing_dti<=.65 and income>=5_000_000: decision="APPROVE"
    else: decision="REVIEW"
    approved_limit=min(float(config.max_limit_vnd),float(offer["credit_limit"])) if decision=="APPROVE" else 0
    expected_loss=pd12*.65*approved_limit
    annual_rate=offer["interest_rate"] if offer["interest_rate"] is not None else 0
    point_reasons=model.reason_codes(row)[0]
    all_reasons=policy_reasons+point_reasons
    st.markdown('<div class="section-title">Decision outcome</div>',unsafe_allow_html=True)
    a,b,c,d,e,f = st.columns(6)
    a.metric("Decision",decision); b.metric("12-month PD", f"{pd12:.2%}"); c.metric("Credit score",score)
    d.metric("Fraud score",f'{fraud_score:.1%}'); e.metric("Approved limit",f'VND {approved_limit/1e6:,.1f}M'); f.metric("Expected loss",f'VND {expected_loss/1e6:,.2f}M')
    st.write("**Decision reason codes:** " + "; ".join(all_reasons))
    p1,p2,p3,p4=st.columns(4)
    p1.metric("Risk band",offer["risk_band"]);p2.metric("Annual rate",f'{annual_rate:.1%}' if annual_rate else "N/A");p3.metric("Policy version",config.policy_version);p4.metric("Model version",meta["model_id"])
    detail=pd.DataFrame(model.points_detail(row)[0]).sort_values("score_points").head(5)
    st.dataframe(detail[["feature","bin","woe","score_points"]],use_container_width=True,hide_index=True)
    st.info(f'Product policy: APPROVE below {config.approve_pd_max:.0%} PD • REVIEW below {config.review_pd_max:.0%} PD • Fraud maximum {config.fraud_max:.0%}')
    audit_record={
        "timestamp_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"application_id":f'DEMO-{datetime.now():%Y%m%d%H%M%S}',
        "product":product,"channel":channel,"model_version":meta["model_id"],"policy_version":config.policy_version,
        "pd_12m":pd12,"credit_score":score,"fraud_score":fraud_score,"requested_amount_vnd":requested_amount,
        "approved_limit_vnd":approved_limit,"decision":decision,"override_status":"NO_OVERRIDE","reason_codes":"; ".join(all_reasons)
    }
    if "decision_audit" not in st.session_state: st.session_state.decision_audit=[]
    b1,b2=st.columns([1,4])
    if b1.button("Record decision",type="primary",use_container_width=True,key="record_decision"):
        st.session_state.decision_audit.insert(0,audit_record)
        st.success(f'Decision {audit_record["application_id"]} recorded in the session audit trail.')
    memo=pd.DataFrame([audit_record]).to_csv(index=False).encode("utf-8")
    b2.download_button("Download decision memo",memo,f'{audit_record["application_id"]}_decision_memo.csv',"text/csv",use_container_width=True,key="download_decision_memo")
    st.markdown('<div class="section-title">Production audit trail</div>',unsafe_allow_html=True)
    if st.session_state.decision_audit:
        audit_df=pd.DataFrame(st.session_state.decision_audit)
        st.dataframe(audit_df,use_container_width=True,hide_index=True,height=260)
        st.download_button("Download audit trail",audit_df.to_csv(index=False).encode("utf-8"),"tnex_decision_audit_trail.csv","text/csv",key="download_audit")
    else:
        st.caption("Record a decision to create an auditable session log with model version, policy version, outcome and reason codes.")
if selected_module == "Model validation":
    st.markdown('<div class="section-title">Independent model validation</div>',unsafe_allow_html=True)
    cols = st.columns(5)
    vals = [("Status",meta["validation_status"]),("OOT AUC",f'{meta["auc"]:.3f}'),("Gini",f'{meta["gini"]:.3f}'),("KS",f'{meta["ks"]:.3f}'),("Brier",f'{meta["brier"]:.3f}')]
    for col,(k,v) in zip(cols,vals): col.metric(k,v)
    st.write("**Reconstruction control:**",payload["reconstruction"]["status"],"| Maximum PD gap:",f'{payload["reconstruction"]["max_model_vs_logit_pd_gap"]:.2e}')
    st.success("Validation covers discrimination, calibration proxy, stability and out-of-time performance. Full evidence is in reports/model_validation_pack.xlsx.")
if selected_module == "Strategy & profitability":
    st.markdown('<div class="section-title">Credit strategy simulator</div>',unsafe_allow_html=True)
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
if selected_module == "Strategy & profitability":
    st.markdown('<div class="section-title">Risk-adjusted profitability</div>',unsafe_allow_html=True)
    pricing_view=pd.DataFrame(payload["v5_profitability"]); st.dataframe(pricing_view,use_container_width=True,hide_index=True)
    st.bar_chart(pricing_view.set_index("risk_band")[["expected_profit","expected_loss"]])
    st.caption("EAD is capped by requested amount and five times monthly income. Expected profit deducts expected loss, funding cost and operating cost; RAROC uses 10% economic capital.")
if selected_module == "Strategy & profitability":
    st.markdown('<div class="section-title">Portfolio stress testing</div>',unsafe_allow_html=True)
    stress=pd.DataFrame(payload["v5_stress_testing"]); st.dataframe(stress,use_container_width=True,hide_index=True)
    s1,s2,s3=st.columns(3)
    for col,scenario in zip([s1,s2,s3],["Base","Downturn","Severe"]):
        r=stress[stress.scenario==scenario].iloc[0]; col.metric(scenario,f'VND {r.expected_profit/1e9:.1f}bn',f'EL {r.expected_loss/1e9:.1f}bn')
    st.bar_chart(stress.set_index("scenario")[["expected_profit","expected_loss"]])
if selected_module == "Model validation":
    st.markdown('<div class="section-title">Walk-forward and champion–challenger</div>',unsafe_allow_html=True)
    wf=pd.DataFrame(payload["v5_champion_challenger"]); st.dataframe(wf,use_container_width=True,hide_index=True)
    st.line_chart(wf.pivot(index="window",columns="model",values="auc"))
    s=payload["v5_summary"]; a,b,c=st.columns(3); a.metric("Champion mean AUC",f'{s["champion_mean_auc"]:.3f}'); b.metric("Challenger mean AUC",f'{s["challenger_mean_auc"]:.3f}'); c.metric("Decision",s["challenger_decision"])
if selected_module == "Production & decisioning":
    st.markdown('<div class="section-title">Production model monitoring</div>',unsafe_allow_html=True)
    mon=pd.DataFrame(payload["v6_delayed_monitoring"])
    mature=mon[mon.label_maturity_rate>=.8].copy(); latest=mon.iloc[-1]; latest_mature=mature.iloc[-1]
    baseline_pd=mon.head(6).mean_pd.mean(); pd_shift=abs(latest.mean_pd/baseline_pd-1)
    prod_decisions=load_csv("production_monitoring_sample.csv")
    approval_by_month=prod_decisions.groupby("month").decision.apply(lambda x:(x=="APPROVE").mean())
    approval_shift=abs(approval_by_month.iloc[-1]-approval_by_month.head(6).mean())
    def traffic_light(value,pass_rule,watch_rule):
        if pass_rule(value): return "PASS"
        if watch_rule(value): return "WATCH"
        return "BREACH"
    alerts=pd.DataFrame([
        ["Score PSI",latest.score_psi,"< 0.10","0.10–0.25",traffic_light(latest.score_psi,lambda x:x<.10,lambda x:x<.25),"Continue" if latest.score_psi<.10 else "Investigate drift"],
        ["Mean PD shift",pd_shift,"< 10%","10%–20%",traffic_light(pd_shift,lambda x:x<.10,lambda x:x<.20),"Continue" if pd_shift<.10 else "Review population mix"],
        ["Approval-rate shift",approval_shift,"< 5pp","5–10pp",traffic_light(approval_shift,lambda x:x<.05,lambda x:x<.10),"Continue" if approval_shift<.05 else "Review policy and channel mix"],
        ["Mature-window AUC",latest_mature.auc_when_mature,"≥ 0.68","0.65–0.68",traffic_light(latest_mature.auc_when_mature,lambda x:x>=.68,lambda x:x>=.65),"Continue" if latest_mature.auc_when_mature>=.68 else "Challenge or recalibrate"],
        ["Mature-window KS",latest_mature.ks_when_mature,"≥ 0.30","0.25–0.30",traffic_light(latest_mature.ks_when_mature,lambda x:x>=.30,lambda x:x>=.25),"Continue" if latest_mature.ks_when_mature>=.30 else "Investigate discrimination"],
        ["Calibration gap",abs(latest_mature.calibration_gap_when_mature),"≤ 2%","2%–3%",traffic_light(abs(latest_mature.calibration_gap_when_mature),lambda x:x<=.02,lambda x:x<=.03),"Continue" if abs(latest_mature.calibration_gap_when_mature)<=.02 else "Recalibrate"]
    ],columns=["Indicator","Current value","Pass threshold","Watch range","Status","Required action"])
    overall="BREACH" if (alerts.Status=="BREACH").any() else ("WATCH" if (alerts.Status=="WATCH").any() else "PASS")
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Applications",f'{mon.applications.sum():,.0f}');m2.metric("Mature months",f'{len(mature)} / {len(mon)}')
    m3.metric("Latest PSI",f'{latest.score_psi:.3f}');m4.metric("Latest mature AUC",f'{latest_mature.auc_when_mature:.3f}');m5.metric("Control status",overall)
    if overall=="BREACH": st.error("One or more monitoring limits have been breached. Initiate investigation and consider rollback or recalibration.")
    elif overall=="WATCH": st.warning("Monitoring remains within hard limits, but one or more indicators require enhanced surveillance.")
    else: st.success("All defined production monitoring indicators are within approved thresholds.")
    st.markdown('<div class="section-title">Threshold-based monitoring alerts</div>',unsafe_allow_html=True)
    st.dataframe(alerts,use_container_width=True,hide_index=True,height=250)
    c1,c2=st.columns([1.2,1])
    with c1: st.line_chart(mon.set_index("month")[["mean_pd","observed_bad_rate","score_psi"]],height=300)
    with c2: st.dataframe(mon[["month","applications","label_maturity_rate","score_psi","status","action"]],use_container_width=True,hide_index=True,height=300)
    st.caption("Outcome-based AUC, KS and calibration controls use the latest mature observation window. Leading indicators are used while newer cohorts season.")
if selected_module == "IFRS 9 ECL":
    stages=pd.DataFrame(payload["v6_ifrs9_stage_summary"])
    scenarios=pd.DataFrame(payload["v6_macro_scenarios"])
    lgd=payload["v6_lgd_validation"]; ead=payload["v6_ead_validation"]
    total_ead=stages.total_ead.sum(); total_ecl=stages.weighted_ecl.sum()
    stage23_accounts=stages.loc[stages.stage!="Stage 1","accounts"].sum()
    stage23_share=stage23_accounts/stages.accounts.sum()
    coverage=total_ecl/total_ead
    e1,e2,e3,e4,e5=st.columns(5)
    e1.metric("Total EAD",f'VND {total_ead/1e9:.1f}B',f'{stages.accounts.sum():,.0f} accounts')
    e2.metric("Weighted ECL",f'VND {total_ecl/1e9:.2f}B',f'{coverage:.1%} coverage')
    e3.metric("Stage 2–3 share",f'{stage23_share:.1%}',f'{stage23_accounts:,.0f} accounts')
    e4.metric("Model validation",f'{lgd["status"]} / {ead["status"]}',"LGD / EAD")
    e5.metric("Build status",payload["v6_summary"]["overall_status"],"IFRS 9 control view")
    st.markdown('<div class="section-title">ECL by impairment stage</div>',unsafe_allow_html=True)
    sleft,sright=st.columns([1.3,1])
    stage_view=stages.copy()
    stage_view["Accounts"]=stage_view["accounts"].map(lambda x:f'{x:,.0f}')
    stage_view["Mean PD"]=stage_view["mean_pd"].map(lambda x:f'{x:.1%}')
    stage_view["Mean LGD"]=stage_view["mean_lgd"].map(lambda x:f'{x:.1%}')
    stage_view["EAD (VND B)"]=stage_view["total_ead"].map(lambda x:f'{x/1e9:.2f}')
    stage_view["ECL (VND B)"]=stage_view["weighted_ecl"].map(lambda x:f'{x/1e9:.2f}')
    stage_view["Coverage"]=stage_view["coverage_ratio"].map(lambda x:f'{x:.1%}')
    with sleft:
        st.dataframe(stage_view[["stage","Accounts","Mean PD","Mean LGD","EAD (VND B)","ECL (VND B)","Coverage"]].rename(columns={"stage":"Stage"}),use_container_width=True,hide_index=True,height=205)
    with sright:
        stage_chart=stages.assign(**{"EAD (VND B)":stages.total_ead/1e9,"ECL (VND B)":stages.weighted_ecl/1e9}).set_index("stage")
        st.bar_chart(stage_chart[["EAD (VND B)","ECL (VND B)"]],height=205)
    st.markdown('<div class="section-title">Forward-looking macroeconomic scenarios</div>',unsafe_allow_html=True)
    mleft,mright=st.columns([1.35,1])
    scenario_view=scenarios.copy()
    scenario_view["Weight"]=scenario_view.weight.map(lambda x:f'{x:.0%}')
    scenario_view["GDP growth"]=scenario_view.gdp_growth.map(lambda x:f'{x:.1%}')
    scenario_view["Unemployment"]=scenario_view.unemployment.map(lambda x:f'{x:.1%}')
    scenario_view["Policy rate"]=scenario_view.policy_rate.map(lambda x:f'{x:.1%}')
    scenario_view["Portfolio PD"]=scenario_view.portfolio_pd.map(lambda x:f'{x:.1%}')
    scenario_view["Portfolio LGD"]=scenario_view.portfolio_lgd.map(lambda x:f'{x:.1%}')
    scenario_view["ECL (VND B)"]=scenario_view.total_ecl.map(lambda x:f'{x/1e9:.2f}')
    with mleft:
        st.dataframe(scenario_view[["scenario","Weight","GDP growth","Unemployment","Policy rate","Portfolio PD","Portfolio LGD","ECL (VND B)"]].rename(columns={"scenario":"Scenario"}),use_container_width=True,hide_index=True,height=205)
    with mright:
        scenario_chart=scenarios.assign(**{"ECL (VND B)":scenarios.total_ecl/1e9}).set_index("scenario")
        st.bar_chart(scenario_chart["ECL (VND B)"],height=205,color="#f59e0b")
    st.markdown('<div class="section-title">LGD and EAD validation</div>',unsafe_allow_html=True)
    v1,v2=st.columns(2)
    with v1:
        st.markdown('<div class="panel-title">Workout LGD proxy</div>',unsafe_allow_html=True)
        l1,l2,l3,l4=st.columns(4)
        l1.metric("Status",lgd["status"]); l2.metric("R²",f'{lgd["r2"]:.3f}'); l3.metric("MAE",f'{lgd["mae"]:.3f}'); l4.metric("Test rows",f'{lgd["test_rows"]:,}')
        lgd_view=pd.DataFrame([[lgd["target"],f'{lgd["mean_observed"]:.2%}',f'{lgd["mean_predicted"]:.2%}',f'{abs(lgd["mean_observed"]-lgd["mean_predicted"]):.2%}']],columns=["Validation target","Observed","Predicted","Calibration gap"])
        st.dataframe(lgd_view,use_container_width=True,hide_index=True)
    with v2:
        st.markdown('<div class="panel-title">EAD factor proxy</div>',unsafe_allow_html=True)
        a1,a2,a3,a4=st.columns(4)
        a1.metric("Status",ead["status"]); a2.metric("R²",f'{ead["r2"]:.3f}'); a3.metric("MAE",f'{ead["mae"]:.3f}'); a4.metric("Test rows",f'{ead["test_rows"]:,}')
        ead_view=pd.DataFrame([[ead["target"],f'{ead["mean_observed"]:.2%}',f'{ead["mean_predicted"]:.2%}',f'{abs(ead["mean_observed"]-ead["mean_predicted"]):.2%}']],columns=["Validation target","Observed","Predicted","Calibration gap"])
        st.dataframe(ead_view,use_container_width=True,hide_index=True)
    with st.expander("Methodology and governance notes"):
        st.markdown("""
        - **Staging:** Stage 1 represents performing exposure; Stage 2 captures significant increase in credit risk; Stage 3 represents credit-impaired exposure.
        - **Forward-looking ECL:** scenario outputs combine PD and LGD adjustments under Upside, Base and Downside conditions using approved probability weights.
        - **Validation:** LGD is tested against discounted loss severity; EAD is tested against balance at default relative to current exposure.
        - **Production control:** figures are portfolio-model outputs for governance testing and require Finance, Model Risk and Credit Committee approval before financial reporting use.
        """)
if selected_module == "Production & decisioning":
    st.markdown('<div class="section-title">Deployment and rollback controls</div>',unsafe_allow_html=True)
    gates=pd.DataFrame(payload["v6_deployment_gates"]);incidents=pd.DataFrame(payload["v6_incidents"]);rollback=pd.DataFrame(payload["v6_rollback_drill"])
    d1,d2,d3=st.columns(3);d1.metric("Rollout status",payload["v6_summary"]["rollout_status"]);d2.metric("Rollback controls",f'{payload["v6_summary"]["rollback_controls_passed"]}/5 PASS');d3.metric("Open incidents",f'{(incidents.status=="OPEN").sum()}')
    st.subheader("Shadow and canary gates");st.dataframe(gates,use_container_width=True,hide_index=True)
    st.subheader("Incident register");st.dataframe(incidents,use_container_width=True,hide_index=True)
    st.subheader("Rollback evidence");st.dataframe(rollback,use_container_width=True,hide_index=True)
if selected_module == "Model validation":
    st.markdown('<div class="section-title">Customer fairness assessment</div>',unsafe_allow_html=True)
    fair=pd.DataFrame(payload["v41_fairness"]); dim=st.selectbox("Assessment dimension",fair.dimension.unique()); view=fair[fair.dimension==dim]
    st.dataframe(view,use_container_width=True,hide_index=True); st.bar_chart(view.set_index("group")[["approval_rate","bad_rate"]]); st.caption("A disparate impact ratio below 0.80 is flagged for review; it does not by itself establish unlawful discrimination.")
if selected_module == "Collections & EWS":
    st.markdown('<div class="section-title">Behavioural early-warning system</div>',unsafe_allow_html=True)
    ews=pd.DataFrame(payload["ews_summary"]); st.dataframe(ews,use_container_width=True,hide_index=True); st.bar_chart(ews.set_index("ews_status")["accounts"])
    accounts=pd.read_csv(ROOT/"data/behavioural_ews_sample.csv"); status=st.selectbox("EWS status",["RED","AMBER","GREEN"]); st.dataframe(accounts[accounts.ews_status==status].head(100),use_container_width=True,hide_index=True)
if selected_module == "Governance & controls":
    st.markdown('<div class="section-title">Manual overrides</div>',unsafe_allow_html=True)
    queue=pd.read_csv(ROOT/"data/manual_review_queue.csv"); st.dataframe(queue[["application_id","pd_12m","decision","override_flag","override_direction","override_reason","final_decision","authority"]].head(200),use_container_width=True,hide_index=True)
    st.metric("Override rate",f'{queue.override_flag.mean():.1%}')
if selected_module == "Governance & controls":
    st.markdown('<div class="section-title">Model governance</div>',unsafe_allow_html=True)
    gov=pd.DataFrame(payload["governance"])
    registry=json.loads((ROOT/"artifacts/model_registry.json").read_text())
    policy=payload["policy"]
    complete=int((gov.status=="COMPLETE").sum())
    ready=int((gov.status=="READY").sum())
    pending=int((gov.status=="PENDING").sum())
    active_version=registry["production"]
    active_record=next((v for v in registry["versions"] if v["version"]==active_version),registry["versions"][-1])
    g1,g2,g3,g4,g5=st.columns(5)
    g1.metric("Governance gates",len(gov),f"{complete} complete")
    g2.metric("Release readiness",f"{complete+ready}/{len(gov)}",f"{pending} pending")
    g3.metric("Active model",active_version,active_record["status"])
    g4.metric("Validation",active_record["metrics"].get("validation_status","N/A"),f'AUC {active_record["metrics"].get("auc",active_record["metrics"].get("oot_auc",0)):.3f}')
    g5.metric("Policy version",policy["policy_version"],f'Effective {policy["effective_date"]}')
    if pending:
        st.warning("Decision required: Model Risk Committee approval and Technology production release remain pending. UAT and monitoring controls are ready.")
    left,right=st.columns([1.35,1])
    with left:
        st.markdown('<div class="section-title">Governance approval path</div>',unsafe_allow_html=True)
        gov_view=gov.rename(columns={"stage":"Gate","owner":"Accountable owner","status":"Status","evidence":"Required evidence"})
        st.dataframe(gov_view,use_container_width=True,hide_index=True,height=330)
    with right:
        st.markdown('<div class="section-title">Approval controls</div>',unsafe_allow_html=True)
        controls=pd.DataFrame([
            ["Independent validation",active_record["metrics"].get("validation_status","N/A"),"Model Risk"],
            ["UAT and reason-code tests",str(gov.loc[gov.stage.str.contains("UAT"),"status"].iloc[0]),"Credit Operations"],
            ["Committee approval",str(gov.loc[gov.stage.str.contains("Committee"),"status"].iloc[0]),"Model Risk Committee"],
            ["Production release",str(gov.loc[gov.stage.str.contains("Production"),"status"].iloc[0]),"Technology"],
            ["Monitoring framework",str(gov.loc[gov.stage.str.contains("Monitoring"),"status"].iloc[0]),"Model Owner"]
        ],columns=["Control","Status","Owner"])
        st.dataframe(controls,use_container_width=True,hide_index=True,height=330)
    st.markdown('<div class="section-title">Model inventory and active credit policy</div>',unsafe_allow_html=True)
    mcol,pcol=st.columns([1.4,1])
    with mcol:
        registry_rows=[]
        for version in registry["versions"]:
            metrics=version.get("metrics",{})
            auc=metrics.get("auc",metrics.get("oot_auc"))
            registry_rows.append({
                "Model version":version["version"],
                "Lifecycle":version["status"],
                "AUC":auc,
                "Gini":metrics.get("gini",metrics.get("test_gini")),
                "KS":metrics.get("ks",metrics.get("test_ks")),
                "Validation":metrics.get("validation_status","N/A"),
                "Production":"YES" if version["version"]==active_version else "NO"
            })
        st.dataframe(pd.DataFrame(registry_rows),use_container_width=True,hide_index=True,height=245)
    with pcol:
        policy_view=pd.DataFrame([
            ["Approve cut-off",f'{policy["approve_pd_max"]:.0%} PD'],
            ["Manual review cut-off",f'{policy["review_pd_max"]:.0%} PD'],
            ["Maximum DTI",f'{policy["maximum_dti"]:.0%}'],
            ["Minimum monthly income",f'VND {policy["minimum_income"]:,.0f}'],
            ["Maximum tenor",f'{policy["maximum_tenor"]} months'],
            ["LGD assumption",f'{policy["lgd"]:.0%}']
        ],columns=["Policy parameter","Approved value"])
        st.dataframe(policy_view,use_container_width=True,hide_index=True,height=245)
    with st.expander("Technical evidence and registry details"):
        t1,t2=st.columns(2)
        with t1:
            st.caption("Model artifact and metric registry")
            st.json(registry,expanded=False)
        with t2:
            st.caption("Machine-readable credit policy")
            st.json(policy,expanded=False)
if selected_module == "Governance & controls":
    st.markdown('<div class="section-title">Consolidated action tracker</div>',unsafe_allow_html=True)
    incidents=load_csv("model_incidents_v6.csv"); findings=pd.DataFrame(payload["v9_validation_findings"])
    actions=pd.concat([
        incidents.assign(item_type="Production incident").rename(columns={"incident_id":"item_id","signal":"issue","sla_hours":"deadline"})[["item_type","item_id","severity","issue","owner","deadline","status","action"]],
        findings.assign(item_type="Validation finding",action="Remediate and validate").rename(columns={"finding_id":"item_id","finding":"issue","due_date":"deadline"})[["item_type","item_id","severity","issue","owner","deadline","status","action"]]
    ],ignore_index=True)
    actions["deadline"]=actions["deadline"].astype(str)
    a,b,c,d=st.columns(4);a.metric("Total actions",len(actions));b.metric("Open",int((actions.status=="OPEN").sum()));c.metric("High/Critical",int(actions.severity.astype(str).str.upper().isin(["HIGH","CRITICAL"]).sum()));d.metric("Owners",actions.owner.nunique())
    status_pick=st.multiselect("Status filter",sorted(actions.status.unique()),default=sorted(actions.status.unique()),key="action_status")
    st.dataframe(actions[actions.status.isin(status_pick)].sort_values(["severity","deadline"]),use_container_width=True,hide_index=True)
    st.info("Action tracker consolidates production incidents and independent-validation findings into one accountable remediation queue.")
if selected_module == "Data governance":
    st.markdown('<div class="section-title">Data quality controls</div>',unsafe_allow_html=True)
    datasets={"Production monitoring":load_csv("production_monitoring_sample.csv"),"TNEX decisions":load_csv("tnex_decisions_v7.csv"),"Behavioural EWS":load_csv("behavioural_ews_sample.csv"),"Manual review":load_csv("manual_review_queue.csv")}
    rows=[]
    for name,df in datasets.items():
        missing=int(df.isna().sum().sum()); cells=max(df.shape[0]*df.shape[1],1)
        duplicate=int(df.duplicated().sum())
        rows.append({"dataset":name,"rows":len(df),"columns":len(df.columns),"completeness":1-missing/cells,"missing_cells":missing,"duplicate_rows":duplicate,"status":"PASS" if missing/cells<.01 and duplicate==0 else "REVIEW"})
    dq=pd.DataFrame(rows)
    a,b,c,d=st.columns(4);a.metric("Datasets checked",len(dq));b.metric("Total rows",f'{sum(len(x) for x in datasets.values()):,}');c.metric("Average completeness",f'{dq.completeness.mean():.2%}');d.metric("Controls passing",f'{(dq.status=="PASS").sum()}/{len(dq)}')
    st.dataframe(dq,use_container_width=True,hide_index=True)
    st.markdown('<div class="section-title">Field-level control checks</div>',unsafe_allow_html=True)
    prod=datasets["Production monitoring"]
    checks=pd.DataFrame([
        ["PD range","0 ≤ pd_12m ≤ 1",prod.pd_12m.between(0,1).mean(),"PASS" if prod.pd_12m.between(0,1).all() else "FAIL"],
        ["Credit score range","300 ≤ score ≤ 850",prod.credit_score.between(300,850).mean(),"PASS" if prod.credit_score.between(300,850).all() else "FAIL"],
        ["Income validity","monthly_income > 0",(prod.monthly_income>0).mean(),"PASS" if (prod.monthly_income>0).all() else "FAIL"],
        ["DTI validity","0 ≤ DTI ≤ 1",prod.existing_dti.between(0,1).mean(),"PASS" if prod.existing_dti.between(0,1).all() else "FAIL"],
        ["Decision domain","Approved values only",prod.decision.isin(["APPROVE","REVIEW","DECLINE"]).mean(),"PASS" if prod.decision.isin(["APPROVE","REVIEW","DECLINE"]).all() else "FAIL"]
    ],columns=["Control","Rule","Pass rate","Status"])
    st.dataframe(checks,use_container_width=True,hide_index=True)
if selected_module == "Data governance":
    st.markdown('<div class="section-title">Business data dictionary</div>',unsafe_allow_html=True)
    dictionary=pd.DataFrame([
        ["application_id","Application","Unique application identifier","String","Origination platform"],
        ["pd_12m","Credit model","Probability of default within 12 months","0–1","PD model"],
        ["credit_score","Credit model","Score mapped monotonically from PD","300–850","Scorecard"],
        ["decision","Decisioning","Policy outcome before manual override","APPROVE/REVIEW/DECLINE","Decision engine"],
        ["existing_dti","Affordability","Existing debt-service-to-income ratio","0–1","Application data"],
        ["requested_amount","Application","Requested principal amount","VND","Application data"],
        ["fraud_score","Fraud","Estimated fraud risk","0–1","Fraud engine"],
        ["behavioural_pd","EWS","Forward-looking behavioural default probability","0–1","Behavioural model"],
        ["days_past_due","Collections","Calendar days payment is overdue","Days","Servicing ledger"],
        ["fpd30","Vintage","First-payment default within 30 days","0/1","Performance mart"],
        ["mob3_30plus","Vintage","30+ DPD observed by month on book 3","0/1","Performance mart"],
        ["score_psi","Monitoring","Population Stability Index for score distribution","Decimal","Monitoring engine"],
        ["weighted_ecl","IFRS 9","Scenario-weighted expected credit loss","VND","ECL engine"],
        ["raroc","Economics","Risk-adjusted return on economic capital","Percentage","Pricing engine"]
    ],columns=["Field","Domain","Business definition","Format","Authoritative source"])
    domain=st.selectbox("Domain",["All"]+sorted(dictionary.Domain.unique().tolist()),key="dictionary_domain")
    if domain!="All": dictionary=dictionary[dictionary.Domain==domain]
    st.dataframe(dictionary,use_container_width=True,hide_index=True,height=520)
    st.caption("Core dictionary for the simulation package. Production implementation should add data owner, lineage, sensitivity classification, retention and refresh SLA.")
if selected_module == "Products & funnel":
    st.markdown('<div class="section-title">TNEX product portfolio</div>',unsafe_allow_html=True)
    s=payload["v7_summary"]; a,b,c,d=st.columns(4); a.metric("Applications",f'{s["applications"]:,}'); b.metric("Products",s["products"]); c.metric("Approval rate",f'{s["approval_rate"]:.1%}'); d.metric("P95 decision",f'{s["p95_end_to_end_ms"]:.0f} ms')
    st.dataframe(pd.DataFrame(payload["v7_product_performance"]),use_container_width=True,hide_index=True)
    st.subheader("Versioned product policy"); st.dataframe(pd.DataFrame(payload["v7_product_config"]),use_container_width=True,hide_index=True)
    st.warning(s["disclaimer"])
if selected_module == "Production & decisioning":
    st.markdown('<div class="section-title">Real-time decisioning and alternative-data controls</div>',unsafe_allow_html=True)
    latency=pd.DataFrame(payload["v7_latency"]); st.bar_chart(latency.set_index("component")["p95_ms"]); st.dataframe(latency,use_container_width=True,hide_index=True)
    st.subheader("Alternative-data governance"); st.dataframe(pd.DataFrame(payload["v7_feature_governance"]),use_container_width=True,hide_index=True)
    st.caption("Consent, freshness and fallback are explicit controls. Missing alternative data never silently becomes an adverse decision.")
if selected_module == "Products & funnel":
    st.markdown('<div class="section-title">Digital application funnel</div>',unsafe_allow_html=True)
    s=payload["v7_summary"]; funnel=pd.DataFrame(payload["v7_funnel"]); st.bar_chart(funnel.set_index("stage")["customers"]); st.dataframe(funnel,use_container_width=True,hide_index=True)
    st.metric("Start-to-disbursement conversion",f'{s["disbursement_conversion"]:.1%}')
if selected_module == "Collections & EWS":
    st.markdown('<div class="section-title">Customer lifecycle controls</div>',unsafe_allow_html=True)
    st.subheader("Repeat-customer limit actions"); st.dataframe(pd.DataFrame(payload["v7_lifecycle"]),use_container_width=True,hide_index=True)
    st.subheader("Payment and collections controls"); st.dataframe(pd.DataFrame(payload["v7_collections"]),use_container_width=True,hide_index=True)
    st.info("Unresolved payment-posting exceptions suppress automated reminders until the ledger is reconciled.")
