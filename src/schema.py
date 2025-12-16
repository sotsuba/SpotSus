from pydantic import BaseModel


class FraudPredictResponse(BaseModel):
    is_fraud: bool
    proba: float
