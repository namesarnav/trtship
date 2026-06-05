"""Turning configured optimization profiles into per-input (min, opt, max) shapes."""

from __future__ import annotations

from collections.abc import Sequence

from trtship.config import OptimizationProfile
from trtship.errors import ConfigError
from trtship.models import ModelSignature

Shape = tuple[int, ...]
ShapeRange = tuple[Shape, Shape, Shape]  # (min, opt, max)
ProfileShapes = dict[str, ShapeRange]


def build_profile_shapes(
    signature: ModelSignature, profiles: Sequence[OptimizationProfile]
) -> list[ProfileShapes]:
    """One ``{input name: (min, opt, max)}`` mapping per optimization profile.

    Static inputs need no entry in a profile: they are given their fixed shape. Every dynamic input
    must appear in every profile. A model with only static inputs needs no profile at all, and an
    empty list is returned when none is configured.
    """
    dynamic = [spec for spec in signature.inputs if spec.is_dynamic]
    if not profiles:
        if dynamic:
            names = ", ".join(spec.name for spec in dynamic)
            raise ConfigError(
                f"input(s) {names} have dynamic dimensions but tensorrt.profiles is empty",
                hint="Add a profile with min/opt/max shapes for each dynamic input.",
            )
        return []

    built: list[ProfileShapes] = []
    for index, profile in enumerate(profiles):
        shapes: ProfileShapes = {}
        for spec in signature.inputs:
            if spec.name in profile.inputs:
                given = profile.inputs[spec.name]
                shapes[spec.name] = (tuple(given.min), tuple(given.opt), tuple(given.max))
            elif spec.is_dynamic:
                raise ConfigError(
                    f"tensorrt.profiles[{index}] has no entry for dynamic input {spec.name!r}"
                )
            else:
                fixed = tuple(d for d in spec.shape if isinstance(d, int))
                shapes[spec.name] = (fixed, fixed, fixed)
        built.append(shapes)
    return built
