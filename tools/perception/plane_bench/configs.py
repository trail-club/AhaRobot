"""Detector configurations (noise model subpixel x detector params x node stride).

``add_arguments`` adds ``--detector-subpixel``, ``--merge``, ``--node-stride``
and ``--param``; ``from_arguments`` expands them into the product grid of
``Config``. ``parse_config`` reads one configuration from
``sub=0.16,merge=on,node_stride=2``.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass

import numpy as np
from aha_perception.planes import DETECTORS, DepthNoiseModel, params_for_input_stride

DEFAULT_SUBPIXEL = DepthNoiseModel().subpixel


@dataclass(frozen=True)
class Config:
    subpixel: float = DEFAULT_SUBPIXEL
    params: tuple = ()  # ((field, value), ...) for the detector params dataclass
    # The node parameter stride (not the detector param ``stride``): the
    # detector gets every node_stride-th row and column of the image, as from
    # depth_to_cloud(..., stride), with params_for_input_stride like the node;
    # intrinsics are not rescaled.
    node_stride: int = 1

    @property
    def label(self) -> str:
        values = dict(self.params)
        parts = [f"sub {self.subpixel:g}"]
        if values.get("merge"):
            parts.append("merge")
        if self.node_stride != 1:
            parts.append(f"node stride {self.node_stride}")
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
        params = DETECTORS[method].params_type(**dict(self.params))
        return params_for_input_stride(params, self.node_stride)

    def decimate(self, image: np.ndarray) -> np.ndarray:
        """The rows and columns of an (H, W, ...) image that the node keeps."""
        return image[:: self.node_stride, :: self.node_stride]


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
        "--node-stride",
        type=int,
        nargs="+",
        default=[1],
        help="node parameter stride: every N-th row and column of the image goes "
        "to the detector; several values run as separate configurations",
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


def parse_config(text: str, methods, node_stride: bool = False) -> Config:
    """``Config`` from ``sub=0.16,merge=on,merge_k=4`` (``sub`` = subpixel).

    ``node_stride=N`` is accepted only by runners that pass ``node_stride=True``.
    """
    subpixel, params, stride = DEFAULT_SUBPIXEL, [], 1
    for item in filter(None, text.split(",")):
        if item.startswith("sub="):
            subpixel = float(item[4:])
        elif item.startswith("node_stride="):
            if not node_stride:
                raise ValueError("node_stride is not supported by this runner")
            stride = int(item[len("node_stride=") :])
            if stride < 1:
                raise ValueError(f"node_stride must be >= 1, got {stride}")
        else:
            params.append(_param(item, methods))
    return Config(subpixel, tuple(params), stride)


def from_arguments(parser, args, methods) -> list[Config]:
    """Product grid of the configuration arguments; errors go to ``parser``."""
    try:
        extra = tuple(_param(item, methods) for item in args.param)
        if "on" in args.merge:
            _param("merge=on", methods)
    except ValueError as error:
        parser.error(f"--param / --merge: {error}")
    if min(args.node_stride) < 1:
        parser.error("--node-stride must be >= 1")
    configs = []
    for subpixel, merge, stride in itertools.product(
        args.detector_subpixel, args.merge, args.node_stride
    ):
        params = (("merge", True),) if merge == "on" else ()
        configs.append(Config(subpixel, params + extra, stride))
    return list(dict.fromkeys(configs))
