"""Rendering of machine-readable reports for people."""

from trtship.reporting.benchmark_report import (
    Comparison,
    compare_reports,
    load_benchmark_reports,
    render_benchmark,
    render_benchmark_markdown,
    render_comparison,
)
from trtship.reporting.engine_report import render_engine_validation
from trtship.reporting.model_report import render_model_report, render_model_report_text
from trtship.reporting.run_report import (
    RunSummary,
    render_run_summary,
    render_run_summary_text,
    summarize_run,
)
from trtship.reporting.serving_report import (
    ServingComparison,
    ServingOverhead,
    compare_serving,
    render_serving,
    render_serving_markdown,
)
from trtship.reporting.validation_report import (
    render_onnx_validation,
    render_onnx_validation_text,
)

__all__ = [
    "Comparison",
    "RunSummary",
    "ServingComparison",
    "ServingOverhead",
    "compare_reports",
    "compare_serving",
    "load_benchmark_reports",
    "render_benchmark",
    "render_benchmark_markdown",
    "render_comparison",
    "render_engine_validation",
    "render_model_report",
    "render_model_report_text",
    "render_onnx_validation",
    "render_onnx_validation_text",
    "render_run_summary",
    "render_run_summary_text",
    "render_serving",
    "render_serving_markdown",
    "summarize_run",
]
