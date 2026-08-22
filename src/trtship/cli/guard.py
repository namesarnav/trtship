"""Error boundary for CLI commands.

Uses only Typer's public API so it behaves identically across Typer/Click versions.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import typer

from trtship.cli.render import render_error, render_unexpected
from trtship.errors import EXIT_UNEXPECTED, ArtifactError, TrtshipError

P = ParamSpec("P")
R = TypeVar("R")


def handle_errors(command: Callable[P, R]) -> Callable[P, R]:
    """Map :class:`TrtshipError` to its documented exit code; unexpected errors exit 70.

    Filesystem errors (:class:`OSError`) are reported as artifact errors, exit code 11, without a
    traceback: they describe the user's environment rather than a defect in trtship.

    ``functools.wraps`` preserves the signature so Typer still introspects the real parameters.
    """

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except (typer.Exit, typer.Abort, typer.BadParameter):
            raise
        except TrtshipError as exc:
            render_error(exc)
            raise typer.Exit(exc.exit_code) from exc
        except OSError as exc:
            if isinstance(exc, ConnectionError | TimeoutError):
                render_unexpected()
                raise typer.Exit(EXIT_UNEXPECTED) from exc
            # A file or directory the user controls (unwritable output, missing input, full disk):
            # an environment problem to report plainly, not a bug to dump a traceback for.
            error = ArtifactError(
                f"{type(exc).__name__}: {exc.strerror or exc}"
                + (f": {exc.filename}" if exc.filename else ""),
                hint="Check that the path exists and is writable, and that the disk has space.",
            )
            render_error(error)
            raise typer.Exit(error.exit_code) from exc
        except Exception as exc:
            render_unexpected()
            raise typer.Exit(EXIT_UNEXPECTED) from exc

    return wrapper
