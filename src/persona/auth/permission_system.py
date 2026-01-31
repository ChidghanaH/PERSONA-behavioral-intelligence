"""Permission-as-Code (PaC) System for PERSONA

This module implements cryptographic permission control that ensures
only authorized users can access PERSONA features and data.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class PermissionScope(Enum):
    """Permission scopes for different PERSONA features"""
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    MODEL_TRAIN = "model:train"
    MODEL_PREDICT = "model:predict"
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"
    USER_MANAGE = "user:manage"
    ORG_ADMIN = "org:admin"
    SYSTEM_ADMIN = "system:admin"


class AccessLevel(Enum):
    """Hierarchical access levels"""
    NONE = 0
    READ_ONLY = 1
    CONTRIBUTOR = 2
    MANAGER = 3
    ADMIN = 4


@dataclass
class PermissionGrant:
    """Represents a cryptographically signed permission grant"""
    grant_id: str
    grantor: str  # Creator or admin who grants permission
    grantee: str  # User or application receiving permission
    resource: str  # Data domain or feature
    scopes: List[str]  # Specific actions allowed
    access_level: str
    issued_at: str
    expires_at: str
    conditions: Dict[str, Any]  # Additional constraints
    signature: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PermissionGrant':
        return cls(**data)


class PermissionManager:
    """Manages creation, validation, and enforcement of permissions"""
    
    def __init__(self, creator_private_key_path: Optional[str] = None):
        self.creator_private_key = None
        self.creator_public_key = None
        self.permission_store: Dict[str, PermissionGrant] = {}
        
        if creator_private_key_path:
            self._load_keys(creator_private_key_path)
        else:
            self._generate_keys()
    
    def _generate_keys(self) -> None:
        """Generate new RSA key pair for creator"""
        logger.info("Generating new RSA key pair for permission system")
        self.creator_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        self.creator_public_key = self.creator_private_key.public_key()
    
    def _load_keys(self, private_key_path: str) -> None:
        """Load existing RSA keys"""
        with open(private_key_path, 'rb') as f:
            self.creator_private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        self.creator_public_key = self.creator_private_key.public_key()
    
    def export_public_key(self) -> bytes:
        """Export public key for distribution"""
        return self.creator_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def create_permission(
        self,
        grantor: str,
        grantee: str,
        resource: str,
        scopes: List[PermissionScope],
        access_level: AccessLevel,
        duration: timedelta = timedelta(days=90),
        conditions: Optional[Dict[str, Any]] = None
    ) -> PermissionGrant:
        """Create a new cryptographically signed permission grant"""
        
        issued_at = datetime.utcnow()
        expires_at = issued_at + duration
        
        # Create grant ID from hash of core components
        grant_data = f"{grantee}|{resource}|{issued_at.isoformat()}"
        grant_id = hashlib.sha256(grant_data.encode()).hexdigest()[:16]
        
        # Prepare payload for signing
        payload = {
            "grant_id": grant_id,
            "grantor": grantor,
            "grantee": grantee,
            "resource": resource,
            "scopes": [s.value for s in scopes],
            "access_level": access_level.name,
            "issued_at": issued_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "conditions": conditions or {}
        }
        
        # Create cryptographic signature
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = self.creator_private_key.sign(
            payload_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        grant = PermissionGrant(
            **payload,
            signature=signature.hex()
        )
        
        # Store grant
        self.permission_store[grant_id] = grant
        logger.info(f"Created permission {grant_id} for {grantee} on {resource}")
        
        return grant
    
    def verify_permission(
        self,
        grant: PermissionGrant,
        public_key: Optional[Any] = None
    ) -> bool:
        """Verify cryptographic signature and validity of permission"""
        
        # Check expiration
        expires_at = datetime.fromisoformat(grant.expires_at)
        if datetime.utcnow() > expires_at:
            logger.warning(f"Permission {grant.grant_id} has expired")
            return False
        
        # Reconstruct payload
        payload = {
            "grant_id": grant.grant_id,
            "grantor": grant.grantor,
            "grantee": grant.grantee,
            "resource": grant.resource,
            "scopes": grant.scopes,
            "access_level": grant.access_level,
            "issued_at": grant.issued_at,
            "expires_at": grant.expires_at,
            "conditions": grant.conditions
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        
        # Verify signature
        try:
            key_to_use = public_key or self.creator_public_key
            key_to_use.verify(
                bytes.fromhex(grant.signature),
                payload_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            logger.debug(f"Permission {grant.grant_id} verified successfully")
            return True
        except Exception as e:
            logger.error(f"Permission verification failed: {e}")
            return False
    
    def check_permission(
        self,
        grantee: str,
        resource: str,
        required_scope: PermissionScope
    ) -> bool:
        """Check if grantee has required permission for resource"""
        
        for grant in self.permission_store.values():
            if (grant.grantee == grantee and 
                grant.resource == resource and
                required_scope.value in grant.scopes):
                
                if self.verify_permission(grant):
                    return True
        
        logger.warning(
            f"Permission denied: {grantee} lacks {required_scope.value} "
            f"on {resource}"
        )
        return False
    
    def revoke_permission(self, grant_id: str) -> bool:
        """Revoke a permission grant"""
        if grant_id in self.permission_store:
            del self.permission_store[grant_id]
            logger.info(f"Revoked permission {grant_id}")
            return True
        return False
    
    def list_permissions(
        self,
        grantee: Optional[str] = None,
        resource: Optional[str] = None
    ) -> List[PermissionGrant]:
        """List all permissions, optionally filtered"""
        grants = list(self.permission_store.values())
        
        if grantee:
            grants = [g for g in grants if g.grantee == grantee]
        if resource:
            grants = [g for g in grants if g.resource == resource]
        
        return grants


class PermissionDecorator:
    """Decorator for enforcing permissions on functions"""
    
    def __init__(self, permission_manager: PermissionManager):
        self.pm = permission_manager
    
    def require_permission(
        self,
        resource: str,
        scope: PermissionScope
    ):
        """Decorator to require specific permission"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract grantee from context (e.g., current user)
                grantee = kwargs.get('user_id') or kwargs.get('grantee')
                if not grantee:
                    raise PermissionError("No user context provided")
                
                if not self.pm.check_permission(grantee, resource, scope):
                    raise PermissionError(
                        f"User {grantee} lacks {scope.value} on {resource}"
                    )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Example usage
if __name__ == "__main__":
    # Initialize permission system
    pm = PermissionManager()
    
    # Create a permission for data analyst
    grant = pm.create_permission(
        grantor="system_admin",
        grantee="analyst_001",
        resource="employee_behavioral_data",
        scopes=[
            PermissionScope.DATA_READ,
            PermissionScope.ANALYTICS_VIEW
        ],
        access_level=AccessLevel.CONTRIBUTOR,
        duration=timedelta(days=90),
        conditions={
            "department": "HR",
            "max_records": 10000,
            "anonymization_required": True
        }
    )
    
    print(f"Created permission: {grant.grant_id}")
    print(f"Valid: {pm.verify_permission(grant)}")
    
    # Check permission
    has_access = pm.check_permission(
        "analyst_001",
        "employee_behavioral_data",
        PermissionScope.DATA_READ
    )
    print(f"Analyst has read access: {has_access}")
