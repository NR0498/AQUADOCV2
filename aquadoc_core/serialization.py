from dataclasses import asdict
from typing import Any, Dict

from .schemas import AnalysisResult


def analysis_result_to_dict(result: AnalysisResult) -> Dict[str, Any]:
    """Convert an analysis result into a stable, JSON-compatible dictionary."""

    return asdict(result)
