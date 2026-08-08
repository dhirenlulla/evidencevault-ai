from typing import Literal
from pydantic import BaseModel


class ComponentHealth(BaseModel):
    """Health information for one infrastructure component."""
    status: Literal["ok", "error"]
    detail: str


class ModelHealth(BaseModel):
    """ 
    Readiness information for a lazily-loaded ML model.

    Deliberately distinct from ComponentHealth: an unloaded
    model is not necessarily an error state - it may simply
    not have been warmed up yet, and will still load lazily on
    first use. Folding this into the same binary status as a
    failed database connection would risk an orchestrator
    killing a perfectly healthy, still-warming instance before
    it ever gets a chance to serve traffic.
    """

    status: Literal["loaded", "not_loaded"]
    model_name: str
    detail: str



class HealthResponse(BaseModel):
    """Complete health response for the EvidenceVault backend."""
    
    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    postgres: ComponentHealth
    qdrant: ComponentHealth
    embedding_model: ModelHealth
    reranker_model: ModelHealth
    
# for understanding -->


# Why use a schema?
# Without a response model, an endpoint can return inconsistent data.

# This schema guarantees that the health response contains:
# status
# service
# version
# environment

# FastAPI also includes this schema in the generated API documentation.