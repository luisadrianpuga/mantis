from enum import Enum


class RiskLevel(str, Enum):
    safe = "safe"
    moderate = "moderate"
    dangerous = "dangerous"
