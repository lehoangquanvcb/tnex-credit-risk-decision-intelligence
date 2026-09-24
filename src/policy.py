from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def load_policy():
    p=yaml.safe_load((ROOT/"config/credit_policy.yaml").read_text());p["effective_date"]=str(p["effective_date"]);return p
def decide(pd12,income,requested,dti=0,tenor=12,policy=None):
    p=policy or load_policy(); fails=[]
    if income<p["minimum_income"]:fails.append("MINIMUM_INCOME")
    if dti>p["maximum_dti"]:fails.append("MAXIMUM_DTI")
    if tenor>p["maximum_tenor"]:fails.append("MAXIMUM_TENOR")
    selected=None
    for name,cfg in p["risk_bands"].items():
        if pd12<cfg["pd_max"]:selected=(name,cfg);break
    if fails or selected is None:return {"policy_version":p["policy_version"],"risk_band":"D","decision":"DECLINE","policy_failures":fails or ["PD_ABOVE_LIMIT"],"interest_rate":None,"credit_limit":0,"expected_loss":0,"expected_profit":0,"raroc":0}
    band,cfg=selected; limit=float(min(requested,income*cfg["income_multiple"])); el=pd12*p["lgd"]*limit; profit=cfg["annual_rate"]*limit-el-p["funding_cost"]*limit-p["operating_cost"]; capital=max(limit*(pd12*p["lgd"]*2+p["economic_capital_floor"]),1)
    return {"policy_version":p["policy_version"],"risk_band":band,"decision":"APPROVE" if pd12<p["approve_pd_max"] else "REVIEW","policy_failures":[],"interest_rate":cfg["annual_rate"],"credit_limit":round(limit),"expected_loss":round(el),"expected_profit":round(profit),"raroc":profit/capital}
