import os
from fastapi import Depends,HTTPException
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
security=HTTPBearer(auto_error=False)
def current_actor(credentials:HTTPAuthorizationCredentials|None=Depends(security)):
    if os.getenv("AUTH_DISABLED","false").lower()=="true":return {"sub":"local-demo","role":"admin"}
    if not credentials:raise HTTPException(401,"Bearer token required")
    try:
        import jwt
        return jwt.decode(credentials.credentials,os.environ["JWT_SECRET"],algorithms=["HS256"],audience="retail-credit-api")
    except Exception as e:raise HTTPException(401,f"Invalid token: {type(e).__name__}")
def require_roles(*roles):
    def check(actor=Depends(current_actor)):
        if actor.get("role") not in roles:raise HTTPException(403,"Insufficient role")
        return actor
    return check
