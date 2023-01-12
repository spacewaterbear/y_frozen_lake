from pydantic import BaseModel


class Params(BaseModel):
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
