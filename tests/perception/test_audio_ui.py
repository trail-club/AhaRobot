"""Mac audio page: session states and HTTP checks, without audio, models or SSH."""

import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perception/macos/audio"))

import audio_ui  # noqa: E402
import session  # noqa: E402

from aha_perception.dgx import RequestError, WorkerError  # noqa: E402


class FakeSegment:
    audio = np.zeros(16000, dtype=np.float32)
    speech_end_time = 0.0


class FakeSegmenter:
    in_speech = False

    def __init__(self):
        self.pending = []

    def push(self, frame, now, muted=False):
        self.muted = muted
        return {"rms": 0.1}, self.pending.pop() if self.pending else None


class FakeWorker:
    def __init__(self, fail=None):
        self.fail = fail
        self.failure = None
        self.closed = threading.Event()
        self.process = None

    def start(self, timeout):
        if self.fail:
            raise self.fail

    def close(self):
        self.closed.set()
        return True


class FakeEngine:
    supports_hotwords = False

    def __init__(self):
        self.error = None

    def transcribe(self, audio, language, hotwords):
        if self.error:
            raise self.error
        return f"{language}:{','.join(hotwords)}"


class FakeTts:
    default_voice = "af_heart"
    voices = {"af_heart": None, "am_adam": None}
    sample_rate = 24000

    def __init__(self):
        self.error = None
        self.used = []

    def synthesize(self, text, voice):
        if self.error:
            raise self.error
        self.used.append(voice)
        yield np.zeros(10, dtype=np.float32)


class FakeMic:
    def __init__(self, device, rate, frame):
        self.device = device

    def start(self):
        pass

    def close(self):
        pass

    def frames(self, stop):
        while not stop.is_set():
            time.sleep(0.01)
            yield np.zeros(512, dtype=np.float32), time.time()


class FakeSpeaker:
    release = None

    def __init__(self, device, rate):
        self.played = 0

    def play(self, audio, rate, cancelled):
        self.played += 1
        if self.release is None:
            return True
        while not self.release.wait(0.01):
            if cancelled():
                return False
        return True

    def close(self):
        pass


class Base(unittest.TestCase):
    def setUp(self):
        self.events = session.Events()
        self.worker = FakeWorker()
        self.engine = FakeEngine()
        self.tts = FakeTts()
        self.segmenter = FakeSegmenter()
        self.calls = []

        def make_worker(target, roles, **kwargs):
            self.calls.append((target, roles, kwargs))
            return self.worker

        def load_engine(registry, key, *args):
            self.calls.append(("load", key))
            return self.engine

        parts = session.Parts(
            make_worker=make_worker,
            load_engine=load_engine,
            make_tts=lambda *args: self.tts,
            mic=FakeMic,
            speaker=FakeSpeaker,
            segmenter=lambda registry: self.segmenter,
            registry=session.Registry(
                path="test",
                raw={
                    "languages": {
                        "en": {"asr": "parakeet_en"},
                        "ja": {"asr": "haas_ja"},
                        "ja-qwen": {"asr": "qwen3_asr"},
                    }
                },
            ),
        )
        self.parts = parts
        self.session = session.Session(session.Options(), self.events, parts)
        self.addCleanup(self.session.close)

    def wait(self, condition, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return
            time.sleep(0.01)
        self.fail(f"timed out; state {self.events.state}")

    def wait_status(self, status):
        self.wait(lambda: self.events.state["status"] == status)

    def connect(self, language="en"):
        self.session.connect("trailpc", language, "taro", "マイク")
        self.wait_status("ready")

    def items(self, kind):
        return [i for i in self.events.history if i["kind"] == kind]


class ConnectionTests(Base):
    def test_connect_loads_the_language_and_starts_listening(self):
        self.connect("ja")
        target, roles, kwargs = self.calls[0]
        self.assertEqual((target, roles), ("trailpc", ["asr", "tts"]))
        self.assertEqual(kwargs["user"], "taro")
        self.assertEqual(self.calls[1], ("load", "haas_ja"))
        self.assertEqual(self.session.link.mic.device, "マイク")
        self.assertEqual(self.events.state["activity"], "listening")

    def test_connect_needs_a_user_unless_an_alias_is_given(self):
        with self.assertRaisesRegex(ValueError, "ユーザー名"):
            self.session.connect("dgx", "en", "", "")
        self.assertEqual(self.events.state["status"], "idle")
        self.session.options.host = "alias"
        self.session.connect("dgx", "en", "", "")
        self.wait_status("ready")

    def test_second_connect_is_refused(self):
        self.connect()
        with self.assertRaisesRegex(ValueError, "切断"):
            self.session.connect("dgx", "en", "taro", "")

    def test_disconnect_closes_the_worker(self):
        self.connect()
        self.session.disconnect()
        self.wait_status("idle")
        self.assertTrue(self.worker.closed.is_set())
        self.assertIsNone(self.session.link)

    def test_ssh_failure_says_what_to_check_and_allows_a_retry(self):
        self.worker.fail = subprocess.CalledProcessError(255, "ssh")
        self.session.connect("dgx", "en", "taro", "")
        self.wait_status("error")
        self.assertIn("DGX Spark", self.events.state["error"])
        self.assertIn("WARP", self.events.state["error"])
        self.assertTrue(self.worker.closed.is_set())
        self.worker = FakeWorker()
        self.connect()
        self.assertEqual(self.events.state["error"], "")

    def test_cancel_while_connecting_is_not_an_error(self):
        started = threading.Event()
        release = threading.Event()
        worker = self.worker

        def slow_start(timeout):
            started.set()
            release.wait(5)

        worker.start = slow_start
        self.session.connect("dgx", "en", "taro", "")
        self.assertTrue(started.wait(5))
        self.session.disconnect()
        release.set()
        self.wait_status("idle")
        self.assertEqual(self.events.state["error"], "")
        self.assertTrue(worker.closed.is_set())

    def test_lost_worker_reports_and_cleans_up(self):
        self.connect()
        self.worker.failure = "ssh ended"
        self.wait_status("error")
        self.assertIn("接続が切れました", self.events.state["error"])
        self.assertTrue(self.worker.closed.is_set())


class ListeningTests(Base):
    def test_segment_becomes_a_heard_item(self):
        self.connect()
        self.session.set_hotwords([" Aha ", " "])
        self.segmenter.pending.append(FakeSegment())
        self.wait(lambda: self.items("heard"))
        self.assertEqual(self.items("heard")[0]["text"], "en:Aha")

    def test_stopped_capture_is_reported_and_cleans_up(self):
        self.connect()

        def broken(frame, now, muted=False):
            raise RuntimeError("onnxruntime failed")

        self.segmenter.push = broken
        self.wait_status("error")
        self.assertIn("音声の処理", self.events.state["error"])
        self.assertTrue(self.worker.closed.is_set())

    def test_failed_request_is_a_notice_and_listening_continues(self):
        self.connect()
        self.engine.error = RequestError("bad audio")
        self.segmenter.pending.append(FakeSegment())
        self.wait(lambda: self.items("notice"))
        self.assertEqual(self.events.state["status"], "ready")
        self.engine.error = None
        self.segmenter.pending.append(FakeSegment())
        self.wait(lambda: self.items("heard"))

    def test_lost_connection_during_recognition_ends_the_session(self):
        self.connect()
        self.engine.error = WorkerError("gone")
        self.segmenter.pending.append(FakeSegment())
        self.wait_status("error")

    def test_language_switch_loads_once_then_applies(self):
        self.connect("en")
        self.session.set_language("ja")
        self.wait(lambda: self.events.state["language"] == "ja")
        self.assertEqual(self.events.state["loading_language"], "")
        self.assertIn(("load", "haas_ja"), self.calls)
        self.segmenter.pending.append(FakeSegment())
        self.wait(lambda: self.items("heard"))
        self.assertEqual(self.items("heard")[0]["language"], "ja")

    def test_qwen_is_a_language_choice(self):
        self.connect("ja")
        self.session.set_language("ja-qwen")
        self.wait(lambda: self.events.state["language"] == "ja-qwen")
        self.assertIn(("load", "qwen3_asr"), self.calls)

    def test_failed_switch_keeps_the_old_language(self):
        self.connect("en")

        def broken(*args):
            raise TimeoutError("slow")

        self.session.parts.load_engine = broken
        self.session.set_language("ja")
        self.wait(lambda: self.items("notice"))
        self.assertEqual(self.events.state["language"], "en")
        self.assertEqual(self.events.state["status"], "ready")


class SpeakingTests(Base):
    def test_speak_records_the_first_audio(self):
        self.connect()
        self.session.speak("Hello. Bye.")
        self.wait(lambda: self.items("said")[0]["status"] == "done")
        self.assertIsNotNone(self.items("said")[0]["first_audio_ms"])
        self.assertEqual(self.session.link.speaker.played, 2)

    def test_stop_cuts_playback_and_a_second_speak_waits(self):
        self.connect()
        FakeSpeaker.release = threading.Event()
        self.addCleanup(setattr, FakeSpeaker, "release", None)
        self.session.speak("Hello. Bye.")
        self.wait(lambda: self.session.link.speaker.played == 1)
        self.assertTrue(self.segmenter.push(None, 0, muted=self.session._muted()))
        self.assertTrue(self.session._muted())
        with self.assertRaisesRegex(ValueError, "待って"):
            self.session.speak("again")
        self.session.stop()
        self.wait(lambda: self.items("said")[0]["status"] == "stopped")
        self.assertEqual(self.session.link.speaker.played, 1)

    def test_failed_sentence_keeps_the_session(self):
        self.connect()
        self.tts.error = TimeoutError("slow")
        self.session.speak("Hello.")
        self.wait(lambda: self.items("said")[0]["status"] == "failed")
        self.assertEqual(self.events.state["status"], "ready")
        self.tts.error = None
        self.wait(lambda: not self.session.speaking.is_set())
        self.session.speak("Again.")

    def test_speak_needs_text_and_a_connection(self):
        with self.assertRaisesRegex(ValueError, "接続"):
            self.session.speak("Hello.")
        self.connect()
        with self.assertRaisesRegex(ValueError, "入力"):
            self.session.speak("  ")


class SpeechOnlyTests(Base):
    def setUp(self):
        super().setUp()
        self.parts.mic = None  # must not be touched
        self.parts.segmenter = None
        self.session = session.Session(
            session.Options(listen=False), self.events, self.parts
        )
        self.addCleanup(self.session.close)

    def connect(self):
        self.session.connect("dgx", "en", "taro", "")
        self.wait_status("ready")

    def test_connects_with_tts_alone_and_lists_the_voices(self):
        self.connect()
        self.assertEqual(self.calls, [("dgx", ["tts"], self.calls[0][2])])
        self.assertIsNone(self.session.link.mic)
        self.assertEqual(self.events.state["voices"], ["af_heart", "am_adam"])
        self.assertEqual(self.events.state["voice"], "af_heart")

    def test_chosen_voice_and_sentence_progress(self):
        self.connect()
        self.session.speak("Hello. Bye.", "am_adam")
        self.wait(lambda: self.items("said")[0]["status"] == "done")
        item = self.items("said")[0]
        self.assertEqual((item["played"], item["sentences"]), (2, 2))
        self.assertEqual(self.tts.used, ["am_adam", "am_adam"])
        self.wait(lambda: self.events.state["activity"] == "")

    def test_unknown_voice_is_refused(self):
        self.connect()
        with self.assertRaisesRegex(ValueError, "声"):
            self.session.speak("Hello.", "nobody")
        self.assertFalse(self.session.speaking.is_set())

    def test_lost_worker_is_noticed_without_listening(self):
        self.connect()
        self.worker.failure = "ssh ended"
        self.wait_status("error")
        self.assertIn("切れました", self.events.state["error"])


class EventsTests(unittest.TestCase):
    def test_new_page_gets_state_and_history(self):
        events = session.Events()
        events.update(status="ready")
        events.add(kind="heard", text="a")
        snapshot = events.subscribe().get_nowait()
        self.assertEqual(snapshot["state"]["status"], "ready")
        self.assertEqual(snapshot["history"][0]["text"], "a")

    def test_stalled_page_does_not_block_the_others(self):
        events = session.Events()
        stalled, live = events.subscribe(), events.subscribe()
        for _ in range(stalled.maxsize + 5):
            events.level(-20.0)
        live.get_nowait()
        self.assertEqual(live.get_nowait()["type"], "level")
        events.unsubscribe(stalled)
        self.assertEqual(len(events.subscribers), 1)


class ServerTests(Base):
    def setUp(self):
        super().setUp()
        defaults = {
            "target": "dgx",
            "language": "en",
            "user": "",
            "user_required": True,
        }
        self.server, self.url = audio_ui.make_server(self.session, defaults, 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, path, body, headers=None):
        request = urllib.request.Request(
            self.url.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers=headers or {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_listens_on_loopback_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_page_and_defaults(self):
        with urllib.request.urlopen(self.url, timeout=5) as response:
            self.assertIn("AhaRobot 音声", response.read().decode())
        for asset in ("page.css", "page.js"):
            with urllib.request.urlopen(self.url + asset, timeout=5) as response:
                self.assertEqual(response.status, 200)
        with urllib.request.urlopen(self.url + "api/defaults", timeout=5) as response:
            self.assertEqual(json.load(response)["target"], "dgx")
        self.assertEqual(
            audio_ui.build_parser().parse_args(["--language", "ja-qwen"]).language,
            "ja-qwen",
        )

    def test_speech_page_has_its_own_port_and_page(self):
        server, url = audio_ui.make_server(self.session, {}, 0, audio_ui.SPEECH_PAGE)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with urllib.request.urlopen(url, timeout=5) as response:
            self.assertIn("AhaRobot 発話", response.read().decode())
        self.assertNotEqual(audio_ui.UI_PORT, audio_ui.SPEECH_UI_PORT)

    def test_connect_through_the_api(self):
        status, _ = self.post(
            "/api/connect",
            {"target": "trailpc", "language": "en", "user": "taro", "device": ""},
        )
        self.assertEqual(status, 200)
        self.wait_status("ready")

    def test_invalid_request_returns_the_reason(self):
        status, body = self.post("/api/speak", {"text": "Hello."})
        self.assertEqual(status, 400)
        self.assertIn("接続", body["error"])
        status, _ = self.post("/api/speak", {})
        self.assertEqual(status, 400)
        status, _ = self.post("/api/unknown", {})
        self.assertEqual(status, 404)

    def test_other_sites_are_refused(self):
        status, _ = self.post(
            "/api/stop",
            {},
            {"Content-Type": "application/json", "Host": "evil.example"},
        )
        self.assertEqual(status, 403)
        status, _ = self.post("/api/stop", {}, {"Content-Type": "text/plain"})
        self.assertEqual(status, 403)

    def test_events_stream_starts_with_a_snapshot(self):
        with urllib.request.urlopen(self.url + "events", timeout=5) as response:
            event = json.loads(response.readline().decode().removeprefix("data: "))
        self.assertEqual(event["type"], "snapshot")


if __name__ == "__main__":
    unittest.main()
