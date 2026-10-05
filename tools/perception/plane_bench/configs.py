"""Detector configurations (noise model subpixel x detector params) for the runners.

``add_arguments`` adds ``--detector-subpixel``, ``--merge`` and ``--param``;
``from_arguments`` expands them into the product grid of ``Config``.
``parse_config`` reads one configuration from ``sub=0.16,merge=on``.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass

from aha_perception.planes import DETECTORS, DepthNoiseModel

DEFAULT_SUBPIXEL = DepthNoiseModel().subpixel


@dataclass(frozen=True)
class Config:
    subpixel: float = DEFAULT_SUBPIXEL
    params: tuple = ()  # ((field, value), ...) for the detector params dataclass

    @property
    def label(self) -> str:
        values = dict(self.params)
        parts = [f"sub {self.subpixel:g}"]
        if values.get("merge"):
            parts.append("merge")
        parts += [
            f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in self.params
            if key != "merge" and (values.get("merge") or not key.startswith("merge_"))
        ]
        return ", ".join(parts)

    @property
    def slug(self) -> str:
        return self.label.replace(", ", "_").replace(" ", "").replace("=", "")

    def noise(self, focal: float) -> DepthNoiseModel:
        return DepthNoiseModel(focal=focal, subpixel=self.subpixel)

    def detector_params(self, method: str):
        return DETECTORS[method].params_type(**dict(self.params))


def _parse_value(text: str, default):
    if isinstance(default, bool):
        if text.lower() in ("1", "true", "on", "yes"):
            return True
        if text.lower() in ("0", "false", "off", "no"):
            return False
        raise ValueError(f"not a bool: {text}")
    return type(default)(text)


def add_arguments(parser) -> None:
    parser.add_argument(
        "--detector-subpixel",
        type=float,
        nargs="+",
        default=[DEFAULT_SUBPIXEL],
        help="subpixel of the detector's DepthNoiseModel (focal = fx); several "
        "values run as separate configurations",
    )
    parser.add_argument(
        "--merge",
        choices=["off", "on"],
        nargs="+",
        default=["off"],
        help="coplanar fragment merging (detector param merge)",
    )
    parser.add_argument(
        "--param",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="detector params field for every configuration, e.g. merge_k=4",
    )


def _param(item: str, methods) -> tuple:
    """(field, value) from ``KEY=VALUE``; every method's params must have KEY."""
    key, sep, text = item.partition("=")
    if not sep:
        raise ValueError(f"expected KEY=VALUE, got {item}")
    default = None
    for method in methods:
        fields = {
            f.name: f.default for f in dataclasses.fields(DETECTORS[method].params_type)
        }
        if key not in fields:
            raise ValueError(f"{method} params have no field {key}")
        default = fields[key]
    return key, _parse_value(text, default)


def parse_config(text: str, methods) -> Config:
    """``Config`` from ``sub=0.16,merge=on,merge_k=4`` (``sub`` = subpixel)."""
    subpixel, params = DEFAULT_SUBPIXEL, []
    for item in filter(None, text.split(",")):
        if item.startswith("sub="):
            subpixel = float(item[4:])
        else:
            params.append(_param(item, methods))
    return Config(subpixel, tuple(params))


def from_arguments(parser, args, methods) -> list[Config]:
    """Product grid of the configuration arguments; errors go to ``parser``."""
    try:
        extra = tuple(_param(item, methods) for item in args.param)
        if "on" in args.merge:
            _param("merge=on", methods)
    except ValueError as error:
        parser.error(f"--param / --merge: {error}")
    configs = []
    for subpixel, merge in itertools.product(args.detector_subpixel, args.merge):
        params = (("merge", True),) if merge == "on" else ()
        configs.append(Config(subpixel, params + extra))
    return list(dict.fromkeys(configs))
