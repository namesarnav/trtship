"""Concrete pipeline stages and the default stage registry."""

from trtship.pipeline.stage import Stage
from trtship.pipeline.stages.build_stage import BuildStage
from trtship.pipeline.stages.calibrate_stage import CalibrateStage
from trtship.pipeline.stages.export_stage import ExportStage
from trtship.pipeline.stages.inspect_stage import InspectStage
from trtship.pipeline.stages.optimize_stage import OptimizeStage
from trtship.pipeline.stages.validate_stage import ValidateStage


def default_stages() -> list[Stage]:
    """Every implemented stage, in canonical order. Later phases append the remaining stages."""
    return [
        InspectStage(),
        ExportStage(),
        ValidateStage(),
        OptimizeStage(),
        CalibrateStage(),
        BuildStage(),
    ]


__all__ = [
    "BuildStage",
    "CalibrateStage",
    "ExportStage",
    "InspectStage",
    "OptimizeStage",
    "ValidateStage",
    "default_stages",
]
