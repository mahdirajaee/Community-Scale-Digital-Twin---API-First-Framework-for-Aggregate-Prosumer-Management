"""
Authentication and Security Module
Provides JWT-based authentication, role-based access control, and security middleware
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.orm import Session
import os
import logging
from functools import wraps

from ..data.database import get_db, User, APIKey
from ..data.models import UserRole, TokenData, UserCreate, UserResponse

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-secret-key-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


class SecurityManager:
    """Centralized security management"""
    
    def __init__(self):
        self.pwd_context = pwd_context
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            
            if username is None:
                return None
                
            return TokenData(username=username, role=UserRole(role) if role else None)
        except JWTError:
            return None
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        user = db.query(User).filter(User.username == username).first()
        
        if not user or not self.verify_password(password, user.hashed_password):
            return None
            
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    def create_user(self, db: Session, user_create: UserCreate) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = db.query(User).filter(User.username == user_create.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Create new user
        hashed_password = self.get_password_hash(user_create.password)
        db_user = User(
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            hashed_password=hashed_password,
            role=user_create.role,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"Created new user: {user_create.username}")
        return db_user


# Initialize security manager
security_manager = SecurityManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not credentials:
        raise credentials_exception
    
    token_data = security_manager.verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_role(required_roles: List[UserRole]):
    """Decorator to require specific roles for endpoint access"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs (assumes it's a dependency)
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if current_user.role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {[role.value for role in required_roles]}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class APIKeyManager:
    """Manage API keys for service-to-service authentication"""
    
    @staticmethod
    def create_api_key(db: Session, name: str, permissions: List[str]) -> APIKey:
        """Create a new API key"""
        import secrets
        
        key = secrets.token_urlsafe(32)
        api_key = APIKey(
            name=name,
            key_hash=security_manager.get_password_hash(key),
            permissions=permissions,
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        # Return the raw key only once
        api_key.raw_key = key
        return api_key
    
    @staticmethod
    def verify_api_key(db: Session, key: str) -> Optional[APIKey]:
        """Verify an API key"""
        api_keys = db.query(APIKey).filter(APIKey.is_active == True).all()
        
        for api_key in api_keys:
            if security_manager.verify_password(key, api_key.key_hash):
                # Update last used
                api_key.last_used = datetime.utcnow()
                db.commit()
                return api_key
        
        return None


async def get_api_key(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[APIKey]:
    """Get API key from request headers"""
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        return None
    
    return APIKeyManager.verify_api_key(db, api_key)


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, identifier: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if request is allowed based on rate limit"""
        current_time = datetime.utcnow()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if (current_time - req_time).seconds < window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= limit:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(requests_per_minute: int = 60):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            
            if not rate_limiter.is_allowed(client_ip, requests_per_minute):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


class SecurityAuditLogger:
    """Log security events for monitoring and compliance"""
    
    @staticmethod
    def log_login_attempt(username: str, success: bool, ip_address: str):
        """Log login attempt"""
        logger.info(f"Login attempt - User: {username}, Success: {success}, IP: {ip_address}")
    
    @staticmethod
    def log_api_access(endpoint: str, method: str, user: str, success: bool):
        """Log API access"""
        logger.info(f"API Access - Endpoint: {endpoint}, Method: {method}, User: {user}, Success: {success}")
    
    @staticmethod
    def log_permission_denied(user: str, endpoint: str, required_role: str):
        """Log permission denied events"""
        logger.warning(f"Permission denied - User: {user}, Endpoint: {endpoint}, Required role: {required_role}")


# Export main components
__all__ = [
    'SecurityManager',
    'security_manager',
    'get_current_user',
    'get_current_active_user',
    'require_role',
    'APIKeyManager',
    'get_api_key',
    'rate_limit',
    'SecurityAuditLogger'
]
