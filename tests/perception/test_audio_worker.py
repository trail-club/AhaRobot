import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "overlay_ws/src/aha_perception")
)

from aha_perception import dgx  # noqa: E402
from aha_perception.audio import asr_engines, audio_io, remote, worker  # noqa: E402
from aha_perception.audio.model_registry import ModelEntry, Registry  # noqa: E402
from aha_perception.audio.tts_engine import split_sentences  # noqa: E402

FAKE_WORKER = (
    Path(__file__).resolve().parents[2]
    / "overlay_ws/src/aha_perception/test/fake_audio_worker.py"
)


class Logger:
    def info(self, text):
        pass


class FakeEngine:
    supports_hotwords = False
    calls = 0

    def __init__(self):
        self.seen = []

    def transcribe(self, audio, language, hotwords):
        self.seen.append((len(audio), language, hotwords))
        return "hello"


class FakeTts:
    voices = {"af_heart": "path"}
    default_voice = "af_heart"
    sample_rate = 24000
    tag = "fake@0000000"

    def __init__(self, registry, device):
        registry.loaded["kokoro"] = {"device": device, "load_seconds": 1.0}

    def synthesize(self, text, voice):
        if text == "boom":
            raise RuntimeError("hf_secret failed")
        yield np.ones(10, dtype=np.float32)
        yield np.ones(5, dtype=np.float32)


def run_worker(requests, roles=("asr", "tts"), loads=None):
    """Serve the requests; returns (exit code, responses)."""
    engine = FakeEngine()

    def load_asr(registry, key, device, logger):
        if loads is not None:
            loads.append(key)
        registry.loaded[key] = {"device": device, "load_seconds": 2.0}
        return engine

    def load():
        return worker.Inference(
            Registry(path="test", raw={}),
            "cuda:0",
            list(roles),
            load_asr=load_asr,
            load_tts=FakeTts,
        )

    output = io.StringIO()
    lines = [r if isinstance(r, str) else json.dumps(r) for r in requests]
    code = worker.serve(io.StringIO("\n".join(lines) + "\n"), output, load)
    return code, [json.loads(line) for line in output.getvalue().splitlines()]


class HaasEngineTests(unittest.TestCase):
    def engine(self):
        module = types.ModuleType("haas_asr")
        module.calls = []
        module.load_model = lambda path, device: (path, device)
        module.Vocabulary = types.SimpleNamespace(parse=tuple)

        def transcribe(model, audios, vocabulary=None):
            module.calls.append((len(audios), vocabulary))
            return ["こんにちは"]

        module.transcribe = transcribe
        entry = ModelEntry("haas_ja", "asr", {"engine": "haas"})
        entry.fetch_file = lambda: "/m/Haas-ja-v1.0.nemo"
        module.decoders = []
        with (
            patch.dict(sys.modules, {"haas_asr": module}),
            patch.object(asr_engines, "without_cuda_graphs", module.decoders.append),
        ):
            return asr_engines.HaasEngine(entry, "cuda:0"), module

    def test_loads_the_pinned_file_on_the_device(self):
        engine, module = self.engine()
        self.assertEqual(engine.model, ("/m/Haas-ja-v1.0.nemo", "cuda:0"))
        self.assertEqual(module.decoders, [engine.model])
        self.assertIn("haas", asr_engines.ENGINES)

    def test_hotwords_become_a_vocabulary_only_when_given(self):
        engine, module = self.engine()
        audio = np.zeros(10, dtype=np.float32)
        self.assertEqual(engine.transcribe(audio, "ja", []), "こんにちは")
        engine.transcribe(audio, "ja", ["Haas=ハース"])
        self.assertEqual(module.calls, [(1, None), (1, ("Haas=ハース",))])


class PcmTests(unittest.TestCase):
    def test_round_trip_is_exact(self):
        audio = np.array([0.0, -1.0, 0.5, 0.125], dtype=np.float32)
        decoded = audio_io.decode_pcm(audio_io.encode_pcm(audio))
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_array_equal(decoded, audio)

    def test_invalid_base64_is_rejected(self):
        with self.assertRaises(ValueError):
            audio_io.decode_pcm("not base64!")


class WorkerTests(unittest.TestCase):
    def test_ready_describes_tts_and_packages(self):
        _, responses = run_worker([])
        ready = responses[0]
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["tts"]["voices"], ["af_heart"])
        self.assertEqual(ready["tts"]["record"]["device"], "cuda:0")
        self.assertIn("packages", ready)

    def test_asr_only_worker_has_no_tts(self):
        code, responses = run_worker(
            [{"type": "synthesize", "id": 1, "text": "x", "voice": "af_heart"}],
            roles=("asr",),
        )
        self.assertEqual(code, 0)
        self.assertNotIn("tts", responses[0])
        self.assertEqual(responses[1]["type"], "error")
        self.assertEqual(responses[1]["id"], 1)

    def test_load_then_transcribe_loads_the_model_once(self):
        loads = []
        audio = audio_io.encode_pcm(np.zeros(1600, dtype=np.float32))
        request = {
            "type": "transcribe",
            "key": "parakeet_en",
            "audio": audio,
            "language": "en",
            "hotwords": [],
        }
        _, responses = run_worker(
            [{"type": "load", "id": 1, "key": "parakeet_en"}, {**request, "id": 2}],
            loads=loads,
        )
        self.assertEqual(loads, ["parakeet_en"])
        self.assertEqual(responses[1]["id"], 1)
        self.assertFalse(responses[1]["supports_hotwords"])
        self.assertEqual(responses[1]["record"]["device"], "cuda:0")
        self.assertEqual(responses[2]["id"], 2)
        self.assertEqual(responses[2]["text"], "hello")

    def test_synthesize_joins_the_chunks_of_a_sentence(self):
        _, responses = run_worker(
            [{"type": "synthesize", "id": 4, "text": "Hi.", "voice": "af_heart"}]
        )
        self.assertEqual(responses[1]["sample_rate"], 24000)
        self.assertEqual(len(audio_io.decode_pcm(responses[1]["audio"])), 15)

    def test_failed_requests_keep_the_worker_running(self):
        _, responses = run_worker(
            [
                "not json",
                {"type": "unknown", "id": 1},
                {"type": "synthesize", "id": 2, "text": "boom", "voice": "af_heart"},
                {"type": "synthesize", "id": 3, "text": "ok", "voice": "af_heart"},
            ]
        )
        kinds = [(r["type"], r.get("id")) for r in responses[1:]]
        self.assertEqual(
            kinds,
            [("error", None), ("error", 1), ("error", 2), ("audio", 3)],
        )
        self.assertNotIn("hf_secret", responses[3]["message"])

    def test_load_failure_is_reported(self):
        def broken():
            raise RuntimeError("no CUDA")

        output = io.StringIO()
        self.assertEqual(worker.serve(io.StringIO(""), output, broken), 1)
        self.assertEqual(json.loads(output.getvalue())["message"], "no CUDA")


class SpecTests(unittest.TestCase):
    def test_targets_choose_the_address(self):
        for target, host in (("dgx", "10.99.0.1"), ("trailpc", "10.99.0.2")):
            worker_ = remote.make_worker(target, ["asr"], user="taro")
            command = worker_.ssh("true")
            self.assertEqual(command[-2], host)
            self.assertEqual(command[command.index("-p") + 1], "2222")
            self.assertIn(target, worker_.spec.title)

    def test_host_overrides_the_address(self):
        worker_ = remote.make_worker("dgx", ["asr"], host="alias", port=0)
        command = worker_.ssh("true")
        self.assertEqual(command[-2], "alias")
        self.assertNotIn("-p", command)

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trailpc"):
            remote.make_worker("elsewhere", ["asr"])

    def test_roles_and_models_reach_the_worker(self):
        spec = remote.audio_spec("dgx", ["asr", "tts"], "cuda:1")
        command = list(spec.command)
        self.assertEqual(command[command.index("--roles") + 1], "asr,tts")
        self.assertEqual(command[command.index("--device") + 1], "cuda:1")
        self.assertEqual(
            command[command.index("--models-file") + 1], f"/app/{remote.MODELS_FILE}"
        )

    def test_sources_exist_and_stay_inside_the_repository(self):
        root = Path(__file__).resolve().parents[2]
        files = remote.audio_sources(root)
        names = {str(path.relative_to(root)) for path in files}
        self.assertTrue(all(path.is_file() for path in files))
        self.assertIn(remote.MODELS_FILE, names)
        self.assertIn(remote.DOCKERFILE, names)
        self.assertIn(
            "overlay_ws/src/aha_perception/aha_perception/audio/worker.py", names
        )
        self.assertFalse(any("upstream" in name for name in names))
        with tempfile.TemporaryFile() as archive:
            dgx.source_archive(files, root, archive)
            self.assertGreater(archive.seek(0, 2), 0)


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.worker = remote.make_worker(
            "trailpc",
            ["asr", "tts"],
            log=lambda text: None,
            command=[sys.executable, str(FAKE_WORKER)],
        )
        self.addCleanup(self.worker.close)
        self.worker.start(30)
        self.registry = Registry(
            path="test", raw={"asr": {"qwen3_asr": {"repo": "q", "revision": "abc"}}}
        )

    def engine(self):
        return remote.load_remote_engine(
            self.registry, "qwen3_asr", self.worker, "trailpc", 10, 10, Logger()
        )

    def test_engine_transcribes_like_a_local_one(self):
        engine = self.engine()
        self.assertTrue(engine.supports_hotwords)
        self.assertEqual(engine.tag, "q@abc")
        self.assertEqual(self.registry.loaded["qwen3_asr"]["device"], "trailpc:cuda:0")
        audio = np.zeros(4800, dtype=np.float32)
        self.assertEqual(engine.transcribe(audio, "ja", ["Aha"]), "ja 3 Aha")

    def test_one_failed_request_keeps_the_connection(self):
        engine = self.engine()
        with self.assertRaises(dgx.RequestError):
            engine.transcribe(np.zeros(8000, dtype=np.float32), "en", [])
        self.assertIsNone(self.worker.failure)
        self.assertEqual(
            engine.transcribe(np.zeros(1600, dtype=np.float32), "en", []), "en 1"
        )

    def test_lost_worker_sets_the_failure(self):
        engine = self.engine()
        with self.assertRaises(dgx.WorkerError):
            engine.transcribe(np.zeros(11200, dtype=np.float32), "en", [])
        self.assertIsNotNone(self.worker.failure)

    def test_tts_returns_the_sentence_audio(self):
        tts = remote.RemoteTts(self.worker, self.registry, "trailpc", 10)
        self.assertEqual(tts.tag, "fake/tts@0000000")
        self.assertIn(tts.default_voice, tts.voices)
        self.assertEqual(self.registry.loaded["kokoro"]["device"], "trailpc:cuda:0")
        chunks = list(tts.synthesize("Hi", tts.default_voice))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 4800)
        with self.assertRaises(dgx.RequestError):
            list(tts.synthesize("fail", tts.default_voice))


class SentenceTests(unittest.TestCase):
    def test_splits_on_punctuation_and_newlines(self):
        self.assertEqual(
            split_sentences("  Hello there. How are you?\nFine: thanks  "),
            ["Hello there.", "How are you?", "Fine:", "thanks"],
        )

    def test_blank_text_has_no_sentences(self):
        self.assertEqual(split_sentences(" \n "), [])


if __name__ == "__main__":
    unittest.main()
