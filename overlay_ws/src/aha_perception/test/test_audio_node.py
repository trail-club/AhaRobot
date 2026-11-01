"""AudioModelNode: a stopped thread loop ends the node."""

import types
from unittest.mock import Mock

from aha_perception.audio.common import AudioModelNode


def fake_node():
    return types.SimpleNamespace(
        loop_failure=None, worker=None, get_logger=lambda: Mock()
    )


def test_exception_in_a_loop_becomes_the_node_error():
    node = fake_node()

    def capture():
        raise RuntimeError("onnxruntime failed")

    AudioModelNode.watched(node, capture)
    assert "capture" in AudioModelNode.error.fget(node)
    assert "onnxruntime failed" in AudioModelNode.error.fget(node)


def test_loop_that_ends_normally_is_no_error():
    node = fake_node()
    AudioModelNode.watched(node, lambda: None)
    assert AudioModelNode.error.fget(node) is None
