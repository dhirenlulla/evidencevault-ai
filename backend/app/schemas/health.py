from typing import Literal
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Structured response returned by the health endpoint"""
    
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    
# for understanding -->


# Why use a schema?
# Without a response model, an endpoint can return inconsistent data.

# This schema guarantees that the health response contains:
# status
# service
# version
# environment

# FastAPI also includes this schema in the generated API documentation.