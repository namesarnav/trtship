"""Rendering of machine-readable reports for people."""

from trtship.reporting.model_report import render_model_report, render_model_report_text
from trtship.reporting.validation_report import (
    render_onnx_validation,
    render_onnx_validation_text,
)

__all__ = [
    "render_model_report",
    "render_model_report_text",
    "render_onnx_validation",
    "render_onnx_validation_text",
]
