from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from schema import FraudPredictResponse

router = APIRouter()


# TODO: Expose actual health status from model, database, etc.
@router.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


# TODO: Implement actual prediction logic, caching, and rate limiting (maybe?)
@router.post("/predict", dependencies=[...])
async def predict() -> FraudPredictResponse:
    return FraudPredictResponse(is_fraud=False, confidence=0.0)


# TODO: Secure this endpoint if needed
@router.get("/metrics")
async def get_metrics():
    headers = {"Content-Type": CONTENT_TYPE_LATEST}
    return Response(content=generate_latest(), headers=headers)
