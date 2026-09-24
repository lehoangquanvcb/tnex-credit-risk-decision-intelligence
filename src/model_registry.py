from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PATH=ROOT/"artifacts/model_registry.json"
def load_registry(): return json.loads(PATH.read_text()) if PATH.exists() else {"production":None,"versions":[]}
def register(version,artifact,metrics,status="VALIDATED"):
    r=load_registry(); r["versions"]=[x for x in r["versions"] if x["version"]!=version]+[{"version":version,"artifact":artifact,"metrics":metrics,"status":status}]; PATH.write_text(json.dumps(r,indent=2)); return r
def promote(version):
    r=load_registry(); assert any(x["version"]==version and x["status"] in ["VALIDATED","STAGING"] for x in r["versions"]); r["previous_production"]=r.get("production"); r["production"]=version; PATH.write_text(json.dumps(r,indent=2)); return r
def rollback():
    r=load_registry(); prev=r.get("previous_production"); assert prev,"No rollback version"; r["production"],r["previous_production"]=prev,r.get("production"); PATH.write_text(json.dumps(r,indent=2)); return r
