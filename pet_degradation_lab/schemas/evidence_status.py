"""Evidence status and scientific classification enumeration."""
from enum import Enum


class EvidenceClassification(str, Enum):
    """
    Mandatory scientific evidence classification.
    Never combine these into a single undifferentiated score.
    """
    EXPERIMENTAL = "EXPERIMENTAL"
    COMPUTED = "COMPUTED"
    PREDICTED = "PREDICTED"
    REPORTED = "REPORTED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    PENDING_VALIDATION = "PENDING_VALIDATION"
