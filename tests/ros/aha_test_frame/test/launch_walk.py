"""Test helper: what `ros2 launch <file> <arguments>` would start, without
starting anything.

``launched`` visits the launch description depth first and in order, as
LaunchService does: arguments are declared, conditions evaluated, and
includes, groups (with their scopes) and OpaqueFunctions executed. The
processes (ExecuteProcess, Node) are collected instead of executed. A Node's
name, namespace, remappings and parameters are evaluated as `ros2 launch`
does it, which writes its parameter dicts to launch_params_* files: these go
to ``tmp_dir``. Processes started only from event handlers are not seen.

aha_test_manipulation's tests load this file from here as well.
"""

from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from launch import LaunchContext
from launch.actions import ExecuteProcess
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node


class _ParamsLoader(yaml.SafeLoader):
    """launch_ros dumps list parameters given as tuples as python/tuple."""


_ParamsLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: loader.construct_sequence(node),
)


@dataclass
class Process:
    label: str  # "<package>/<executable>" of a Node, else the command line
    cmd: list = field(default_factory=list)  # ExecuteProcess only
    package: str | None = None
    executable: str | None = None
    name: str | None = None  # node name without the namespace, if given
    namespace: str | None = None  # if given
    remappings: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)  # from dicts and -p rules
    parameter_files: list = field(default_factory=list)  # given as files


def load_launch(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launched(path, tmp_dir, **arguments) -> list[Process]:
    """The processes `ros2 launch <path> <name>:=<value>...` would start."""
    context = LaunchContext()
    context.launch_configurations.update(arguments)
    out = []

    def walk(entity):
        if isinstance(entity, ExecuteProcess):
            if entity.condition is None or entity.condition.evaluate(context):
                out.append(_process(entity, context, Path(tmp_dir)))
            return
        for sub in entity.visit(context) or []:
            walk(sub)

    saved = tempfile.tempdir
    tempfile.tempdir = str(tmp_dir)
    try:
        walk(load_launch(path).generate_launch_description())
    finally:
        tempfile.tempdir = saved
    return out


def _text(context, value) -> str:
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def _process(entity, context, tmp_dir: Path) -> Process:
    if not isinstance(entity, Node):
        cmd = [_text(context, part) for part in entity.cmd]
        return Process(label=" ".join(cmd), cmd=cmd)
    # launch_ros (Jazzy) private API: the node's evaluated arguments.
    entity._perform_substitutions(context)
    parameters, files = {}, []
    # (argument, is_file) as the node's command line gets them.
    for argument, is_file in entity._Node__expanded_parameter_arguments or []:
        if not is_file:
            name, value = argument.split(":=", 1)
            parameters[name] = yaml.safe_load(value)
        elif Path(argument).parent == tmp_dir:
            (body,) = yaml.load(Path(argument).read_text(), _ParamsLoader).values()
            parameters.update(body["ros__parameters"])
        else:
            files.append(argument)
    name = entity.node_name.rpartition("/")[2]
    return Process(
        label=f"{entity.node_package}/{entity.node_executable}",
        package=entity.node_package,
        executable=entity.node_executable,
        name=None if name == Node.UNSPECIFIED_NODE_NAME else name,
        namespace=(
            None
            if entity.expanded_node_namespace == Node.UNSPECIFIED_NODE_NAMESPACE
            else entity.expanded_node_namespace
        ),
        remappings=dict(entity.expanded_remapping_rules or []),
        parameters=parameters,
        parameter_files=files,
    )
