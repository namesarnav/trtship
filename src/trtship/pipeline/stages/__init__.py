"""Concrete pipeline stages and the default stage registry."""

from trtship.pipeline.stage import Stage
from trtship.pipeline.stages.export_stage import ExportStage
from trtship.pipeline.stages.inspect_stage import InspectStage
from trtship.pipeline.stages.optimize_stage import OptimizeStage
from trtship.pipeline.stages.validate_stage import ValidateStage


def default_stages() -> list[Stage]:
    """Every implemented stage, in canonical order. Later phases append the engine stages."""
    return [InspectStage(), ExportStage(), ValidateStage(), OptimizeStage()]


__all__ = [
    "ExportStage",
    "InspectStage",
    "OptimizeStage",
    "ValidateStage",
    "default_stages",
]
