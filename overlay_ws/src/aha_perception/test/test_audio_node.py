"""AudioModelNode: a stopped thread loop ends the node. AsrNode: Listen deadlines."""

import threading
import time
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


def test_listen_outlasts_a_model_load_past_the_hard_deadline():
    """The hard deadline (timeout + max_segment_s + 5 s) does not count model loading."""
    from aha_msgs.msg import SpeechTranscript

    from aha_perception.audio.asr_node import AsrNode, _ListenRequest

    req = _ListenRequest("ja-qwen", [])
    node = types.SimpleNamespace(
        _request=req,
        _request_lock=threading.Lock(),
        _recognizing=True,
        _loading=True,
        segmenter=types.SimpleNamespace(
            cfg=types.SimpleNamespace(max_segment_s=0.0), in_speech=False
        ),
    )

    def load_then_hear():
        time.sleep(6.5)  # longer than the 5.2 s hard deadline
        node._loading = node._recognizing = False
        req.transcript = SpeechTranscript(text="hi")
        req.done.set()

    threading.Thread(target=load_then_hear, daemon=True).start()
    goal = Mock(is_cancel_requested=False, request=types.SimpleNamespace(timeout=0.2))
    result = AsrNode._execute_listen(node, goal)
    assert result.success
    assert result.transcript.text == "hi"
    goal.succeed.assert_called_once()
    assert node._request is None
