from __future__ import annotations
import json, os, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def log_decision(application_id,model_id,inputs,outputs,user_id="api",override=None):
    record=(str(uuid.uuid4()),application_id,model_id,json.dumps(inputs),json.dumps(outputs),json.dumps(override) if override else None,user_id,datetime.now(timezone.utc).isoformat())
    url=os.getenv("DATABASE_URL")
    if url and url.startswith("postgresql"):
        import psycopg
        from psycopg.types.json import Jsonb
        with psycopg.connect(url) as db:
            db.execute("INSERT INTO scoring_audit (audit_id,application_id,model_id,input_snapshot,output_snapshot,override_snapshot,user_id,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",(record[0],application_id,model_id,Jsonb(inputs),Jsonb(outputs),Jsonb(override) if override else None,user_id,record[7]))
        return record[0]
    db=sqlite3.connect(ROOT/"artifacts/scoring_audit_demo.db"); db.execute("CREATE TABLE IF NOT EXISTS scoring_audit (audit_id TEXT PRIMARY KEY, application_id TEXT, model_id TEXT, input_snapshot TEXT, output_snapshot TEXT, override_snapshot TEXT, user_id TEXT, created_at TEXT)"); db.execute("INSERT INTO scoring_audit VALUES (?,?,?,?,?,?,?,?)",record); db.commit(); db.close(); return record[0]
def apply_override(audit_id,override,user_id):
    url=os.getenv("DATABASE_URL");payload=dict(override,approved_by=user_id,approved_at=datetime.now(timezone.utc).isoformat())
    if url and url.startswith("postgresql"):
        import psycopg
        from psycopg.types.json import Jsonb
        with psycopg.connect(url) as db: result=db.execute("UPDATE scoring_audit SET override_snapshot=%s WHERE audit_id=%s",(Jsonb(payload),audit_id))
        return result.rowcount
    db=sqlite3.connect(ROOT/"artifacts/scoring_audit_demo.db");result=db.execute("UPDATE scoring_audit SET override_snapshot=? WHERE audit_id=?",(json.dumps(payload),audit_id));db.commit();count=result.rowcount;db.close();return count
def find_audit(application_id=None,limit=100):
    db=sqlite3.connect(ROOT/"artifacts/scoring_audit_demo.db");db.row_factory=sqlite3.Row
    q="SELECT audit_id,application_id,model_id,output_snapshot,override_snapshot,user_id,created_at FROM scoring_audit";args=[]
    if application_id:q+=" WHERE application_id=?";args=[application_id]
    q+=" ORDER BY created_at DESC LIMIT ?";args.append(min(limit,500));rows=[dict(x) for x in db.execute(q,args).fetchall()];db.close();return rows
