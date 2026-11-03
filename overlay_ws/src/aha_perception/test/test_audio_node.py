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


def test_mic_unmutes_when_the_tts_node_is_gone():
    """A TTS node that dies mid-speech never publishes tts_active=False."""
    from aha_perception.audio.asr_node import AsrNode

    publishers = [1]
    node = types.SimpleNamespace(
        _tts_active=True,
        _muted_until=0.0,
        count_publishers=lambda topic: publishers[0],
        get_logger=lambda: Mock(),
    )
    assert AsrNode._muted(node)
    publishers[0] = 0
    assert not AsrNode._muted(node)
    assert node._tts_active is False


def test_speech_from_before_the_goal_does_not_answer_it():
    """Segments left in the queue (e.g. while a model loaded) predate the goal."""
    import queue

    import numpy as np

    from aha_perception.audio.asr_node import AsrNode, _ListenRequest

    req = _ListenRequest("ja", ["Haas=ハース"])
    heard = []

    class Engine:
        tag = "fake"
        supports_hotwords = True

        def transcribe(self, audio, language, hotwords):
            heard.append((language, hotwords))
            return f"said {len(heard)}"

    def segment(end):
        return types.SimpleNamespace(
            audio=np.zeros(1600, dtype=np.float32),
            speech_end_time=end,
            speech_seconds=0.1,
            end_reason="turn",
            turn_prob=0.9,
            endpoint_delay=0.1,
        )

    segments = queue.Queue()
    segments.put((1, segment(req.accepted - 2.0)))  # finished before the goal
    segments.put((2, segment(req.accepted + 0.5)))
    node = types.SimpleNamespace(
        _stop=threading.Event(),
        _segments=segments,
        _request=req,
        _request_lock=threading.Lock(),
        _recognizing=False,
        language="en",
        _hotwords=lambda extra: list(extra),
        _save_segment=lambda i, s: "",
        _engine_for=lambda lang: Engine(),
        registry=types.SimpleNamespace(language_code=lambda lang: lang),
        pub_transcript=Mock(),
        debug=Mock(),
        get_logger=lambda: Mock(),
    )
    thread = threading.Thread(target=AsrNode._asr_loop, args=(node,), daemon=True)
    thread.start()
    assert req.done.wait(5)
    node._stop.set()
    thread.join(5)
    assert heard == [("en", []), ("ja", ["Haas=ハース"])]
    assert req.transcript.text == "said 2"


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
