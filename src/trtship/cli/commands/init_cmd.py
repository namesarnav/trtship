"""`trtship init`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.render import console
from trtship.errors import ArtifactError, ConfigError
from trtship.utils.fs import atomic_write_text

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FACTORY = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")

TEMPLATE = """\
# trtship configuration. Validate it with:  trtship config validate {filename}
seed: 0

model:
  name: {name}
  kind: module             # module | checkpoint | torchscript
  factory: {factory}       # package.module:callable that returns an nn.Module
  # python_path: [.]       # directories added to sys.path before importing the factory
  # path: weights.pt       # required for kind: checkpoint and kind: torchscript
  # trust_source: false    # torchscript archives and full pickles execute code when loaded
  inputs:
    - name: input
      dtype: float32
      shape: [batch, 3, 224, 224]   # a string dimension is dynamic
  # output_names: [logits]

export:
  opset: 17

optimize:
  enabled: true

tensorrt:
  precisions: [fp32]       # fp32 | fp16 | int8 (int8 also needs a calibration section)
  workspace_mb: 4096
  profiles:                # required for dynamic inputs
    - inputs:
        input:
          min: [1, 3, 224, 224]
          opt: [8, 3, 224, 224]
          max: [16, 3, 224, 224]

validation:
  num_samples: 8

artifacts:
  root: runs
"""


def init_command(
    path: Annotated[Path, typer.Argument(help="Where to write the config.")] = Path("trtship.yaml"),
    name: Annotated[
        str, typer.Option(help="Model name (used for artifact and Triton names).")
    ] = "my-model",
    factory: Annotated[
        str, typer.Option(help="package.module:callable returning your nn.Module.")
    ] = "my_package.models:build",
    force: Annotated[bool, typer.Option(help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter configuration to edit."""
    if not _NAME.match(name):
        raise ConfigError(
            f"invalid model name {name!r}", hint="Use letters, digits, '_', '.', '-'."
        )
    if not _FACTORY.match(factory):
        raise ConfigError(f"invalid factory {factory!r}", hint="Use package.module:callable.")
    if path.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {path}", hint="Pass --force to overwrite it."
        )
    atomic_write_text(path, TEMPLATE.format(filename=path.name, name=name, factory=factory))
    console.print(f"wrote {path}")
    console.print(f"next: edit the model section, then `trtship config validate {path}`")
