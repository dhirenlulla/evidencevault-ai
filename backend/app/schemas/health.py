from typing import Literal
from pydantic import BaseModel


class ComponentHealth(BaseModel):
    """Health information for one infrastructure component."""
    status: Literal["ok", "error"]
    detail: str
    


class HealthResponse(BaseModel):
    """Complete health response for the EvidenceVault backend."""
    
    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    postgres: ComponentHealth
    qdrant: ComponentHealth
    
# for understanding -->


# Why use a schema?
# Without a response model, an endpoint can return inconsistent data.

# This schema guarantees that the health response contains:
# status
# service
# version
# environment

# FastAPI also includes this schema in the generated API documentation.