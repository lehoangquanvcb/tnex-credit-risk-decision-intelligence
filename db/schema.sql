CREATE TABLE IF NOT EXISTS scoring_audit (
  audit_id UUID PRIMARY KEY,
  application_id VARCHAR(64) NOT NULL,
  model_id VARCHAR(32) NOT NULL,
  input_snapshot JSONB NOT NULL,
  output_snapshot JSONB NOT NULL,
  override_snapshot JSONB,
  user_id VARCHAR(128) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scoring_audit_application ON scoring_audit(application_id);
CREATE INDEX IF NOT EXISTS idx_scoring_audit_created_at ON scoring_audit(created_at);
