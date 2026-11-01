"""One conversation with the audio models on a GPU host; the browser page drives it.

Session owns the worker, microphone and speaker. Everything the page shows is
published through Events, so the page can reconnect at any time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "overlay_ws/src/aha_perception"
sys.path.insert(0, str(PACKAGE))

from aha_perception.audio.audio_io import SAMPLE_RATE, MicStream, Speaker  # noqa: E402
from aha_perception.audio.model_registry import Registry  # noqa: E402
from aha_perception.audio.remote import (  # noqa: E402
    SSH_PORT,
    RemoteTts,
    load_remote_engine,
    make_worker,
)
from aha_perception.audio.segmenter import (  # noqa: E402
    FRAME,
    Segmenter,
    SegmenterConfig,
    SileroVad,
    SmartTurn,
)
from aha_perception.audio.tts_engine import split_sentences  # noqa: E402
from aha_perception.dgx import RequestError  # noqa: E402

MUTE_TAIL_S = 0.3  # keep ignoring the microphone after playback (reverberation)
LEVEL_INTERVAL_S = 0.1
HISTORY = 200
TARGET_NAMES = {"dgx": "DGX Spark", "trailpc": "trailpc"}
LANGUAGE_NAMES = {"en": "英語", "ja": "日本語（Haas）", "ja-qwen": "日本語（Qwen）"}


class Logger:
    def info(self, text):
        print(f"[audio] {text}", flush=True)


class Events:
    """State and history for new pages, and a stream of changes for open ones."""

    def __init__(self):
        self.lock = threading.Lock()
        self.subscribers: list[queue.Queue] = []
        self.state: dict = {}
        self.history: deque = deque(maxlen=HISTORY)
        self.next_id = 1

    def subscribe(self) -> queue.Queue:
        subscriber = queue.Queue(maxsize=500)
        with self.lock:
            subscriber.put(
                {
                    "type": "snapshot",
                    "state": dict(self.state),
                    "history": list(self.history),
                }
            )
            self.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    def _send(self, event: dict) -> None:
        for subscriber in self.subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass  # a stalled page catches up from the next snapshot

    def update(self, **changes) -> None:
        with self.lock:
            self.state.update(changes)
            self._send({"type": "state", "state": changes})

    def level(self, db: float) -> None:
        with self.lock:
            self._send({"type": "level", "db": db})

    def add(self, **item) -> dict:
        with self.lock:
            item["id"] = self.next_id
            self.next_id += 1
            self.history.append(item)
            self._send({"type": "item", "item": item})
        return item

    def change(self, item: dict, **changes) -> None:
        with self.lock:
            item.update(changes)
            self._send({"type": "item", "item": item})


@dataclass
class Options:
    models_file: str = str(PACKAGE / "config/audio/models.yaml")
    host: str = ""  # SSH alias; empty: the address of the target
    port: int = SSH_PORT
    identity_file: str = ""
    output_device: str = ""
    capture_rate: int = SAMPLE_RATE
    startup_timeout: float = 1800.0
    timeout: float = 30.0
    listen: bool = True  # False: speech output only, without microphone or ASR


def new_segmenter(registry):
    import torch

    torch.set_num_threads(1)
    return Segmenter(SileroVad(registry), SmartTurn(registry), SegmenterConfig())


@dataclass
class Parts:
    """What Session builds on; tests replace these."""

    make_worker: object = make_worker
    load_engine: object = load_remote_engine
    make_tts: object = RemoteTts
    mic: object = MicStream
    speaker: object = Speaker
    segmenter: object = new_segmenter
    registry: Registry | None = None


def input_devices() -> list[str]:
    try:
        import sounddevice as sd

        return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception:  # no PortAudio: the page falls back to the default device
        return []


def explain(target: str, error: Exception) -> str:
    """What happened and what to do next, for the page."""
    name = TARGET_NAMES.get(target, target)
    if isinstance(error, subprocess.CalledProcessError):
        return (
            f"{name} に接続できませんでした。"
            "WARPの接続とユーザー名を確認してから、もう一度接続してください。"
        )
    if isinstance(error, TimeoutError):
        return f"{name} から時間内に応答がありませんでした。もう一度接続してください。"
    return f"{name} との接続で問題が起きました: {error}"


@dataclass
class Link:
    """Everything that lives while connected."""

    target: str
    worker: object
    stop: threading.Event = field(default_factory=threading.Event)
    engines: dict = field(default_factory=dict)
    tts: object = None
    mic: object = None
    speaker: object = None
    threads: list = field(default_factory=list)
    segments: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=8))


class Session:
    def __init__(self, options: Options, events: Events, parts: Parts | None = None):
        self.options = options
        self.events = events
        self.parts = parts or Parts()
        self.registry = self.parts.registry or Registry.load(options.models_file)
        self.lock = threading.Lock()
        self.link: Link | None = None
        self.segmenter = None
        self.cancelled = threading.Event()
        self.speaking = threading.Event()
        self.stop_speech = threading.Event()
        self.muted_until = 0.0
        self.recognizing = False
        self.events.update(
            status="idle",
            step="",
            error="",
            target="",
            language="en",
            loading_language="",
            activity="",
            hotwords_applied=False,
            hotwords=[],
            voices=[],
            voice="",
            devices=input_devices(),
        )

    # -- Connection --------------------------------------------------------

    def connect(self, target: str, language: str, user: str, device: str) -> None:
        """Start connecting in the background; ValueError if that cannot start."""
        if language not in LANGUAGE_NAMES:
            raise ValueError("言語を選んでください")
        if not user and not self.options.host:
            raise ValueError("ユーザー名を入力してください")
        with self.lock:
            if self.events.state["status"] in ("connecting", "ready", "closing"):
                raise ValueError("接続済みです。切断してから接続してください")
            self.cancelled.clear()
            self.events.update(
                status="connecting",
                target=target,
                language=language,
                error="",
                step="接続を準備しています",
            )
        threading.Thread(
            target=self._connect, args=(target, user, device), daemon=True
        ).start()

    def _step(self, text: str) -> None:
        self.events.update(step=text)

    def _connect(self, target: str, user: str, device: str) -> None:
        o = self.options
        logger = Logger()
        link = None
        try:
            if o.listen and self.segmenter is None:
                self.segmenter = self.parts.segmenter(self.registry)
            name = TARGET_NAMES.get(target, target)
            self._step(f"{name} でコンテナを起動しています（初回は数十分かかります）")
            worker = self.parts.make_worker(
                target,
                ["asr", "tts"] if o.listen else ["tts"],
                host=o.host,
                port=o.port,
                user=user,
                identity_file=o.identity_file,
                log=logger.info,
            )
            link = self.link = Link(target=target, worker=worker)
            worker.start(o.startup_timeout)
            self._check_cancelled()
            if o.listen:
                language = self.events.state["language"]
                self._step(f"{LANGUAGE_NAMES[language]}の認識モデルを読み込んでいます")
                self._load(link, language)
                self._check_cancelled()
            link.tts = self.parts.make_tts(worker, self.registry, target, o.timeout)
            link.speaker = self.parts.speaker(o.output_device, link.tts.sample_rate)
            loops = [self._watch]
            ready = {"activity": ""}
            if o.listen:
                link.mic = self.parts.mic(device, o.capture_rate, FRAME)
                loops += [self._capture, self._recognize]
                ready = {
                    "activity": "listening",
                    **self._hotword_state(link, language),
                }
            for loop in loops:
                thread = threading.Thread(target=loop, args=(link,), daemon=True)
                link.threads.append(thread)
                thread.start()
            if o.listen:
                link.mic.start()
            self.events.update(
                status="ready",
                step="",
                voices=sorted(link.tts.voices),
                voice=link.tts.default_voice,
                **ready,
            )
        except Exception as exc:
            print(f"[audio] {exc}", file=sys.stderr, flush=True)
            cancelled = self.cancelled.is_set()
            self._close(link)
            self.events.update(
                status="idle" if cancelled else "error",
                step="",
                error="" if cancelled else explain(target, exc),
            )

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise RuntimeError("disconnected")

    def _load(self, link: Link, language: str) -> None:
        key = self.registry.language(language)["asr"]
        if key not in link.engines:
            link.engines[key] = self.parts.load_engine(
                self.registry,
                key,
                link.worker,
                link.target,
                self.options.startup_timeout,
                self.options.timeout,
                Logger(),
            )

    def _engine(self, link: Link, language: str):
        return link.engines[self.registry.language(language)["asr"]]

    def _hotword_state(self, link: Link, language: str) -> dict:
        engine = self._engine(link, language)
        return {"hotwords_applied": bool(engine.supports_hotwords)}

    def disconnect(self) -> None:
        with self.lock:
            status = self.events.state["status"]
            link = self.link
            if status == "connecting":
                self.cancelled.set()
                self._step("接続を中止しています")
                # Ends a worker that is loading; a build in progress stops at
                # the next step of _connect.
                process = getattr(link.worker, "process", None) if link else None
                if process is not None:
                    process.terminate()
                return
            if status != "ready":
                return
            self.events.update(status="closing", step="推論先を片付けています")
        threading.Thread(target=self._shutdown, args=("idle", ""), daemon=True).start()

    def _shutdown(self, status: str, error: str) -> None:
        self._close(self.link)
        self.events.update(status=status, step="", error=error, activity="")

    def _close(self, link: Link | None) -> None:
        if link is None:
            return
        link.stop.set()
        self.stop_speech.set()
        current = threading.current_thread()
        for thread in link.threads:
            if thread is not current:
                thread.join(timeout=2)
        for device in (link.mic, link.speaker):
            try:
                if device is not None:
                    device.close()
            except Exception as exc:
                print(f"[audio] {exc}", file=sys.stderr, flush=True)
        if not link.worker.close():
            print("[audio] Worker cleanup was incomplete", file=sys.stderr, flush=True)
        self.link = None

    def close(self) -> None:
        self.cancelled.set()
        self._close(self.link)

    def _lost(self, link: Link, reason: str) -> None:
        with self.lock:
            if self.events.state["status"] != "ready":
                return
            self.events.update(status="closing", step="推論先を片付けています")
        name = TARGET_NAMES.get(link.target, link.target)
        print(f"[audio] {reason}", file=sys.stderr, flush=True)
        self._shutdown(
            "error", f"{name} との接続が切れました。もう一度接続してください。"
        )

    def _watch(self, link: Link) -> None:
        while not link.stop.wait(0.2):
            if link.worker.failure is not None:
                self._lost(link, link.worker.failure)
                return

    # -- Listening ---------------------------------------------------------

    def _muted(self) -> bool:
        return self.speaking.is_set() or time.time() < self.muted_until

    def _activity(self) -> None:
        if self.speaking.is_set():
            activity = "speaking"
        elif self.recognizing:
            activity = "recognizing"
        elif self.segmenter.in_speech:
            activity = "hearing"
        else:
            activity = "listening"
        if activity != self.events.state.get("activity"):
            self.events.update(activity=activity)

    def _capture(self, link: Link) -> None:
        peak = 0.0
        last = time.time()
        for frame, now in link.mic.frames(link.stop):
            info, segment = self.segmenter.push(frame, now, muted=self._muted())
            peak = max(peak, info["rms"])
            if segment is not None:
                try:
                    link.segments.put_nowait(segment)
                except queue.Full:
                    print("[audio] ASR is falling behind", flush=True)
            if now - last >= LEVEL_INTERVAL_S:
                self.events.level(round(20 * math.log10(max(peak, 1e-5)), 1))
                self._activity()
                peak, last = 0.0, now

    def _recognize(self, link: Link) -> None:
        while not link.stop.is_set():
            try:
                segment = link.segments.get(timeout=0.2)
            except queue.Empty:
                continue
            language = self.events.state["language"]
            self.recognizing = True
            started = time.time()
            try:
                text = self._engine(link, language).transcribe(
                    segment.audio, language, self.events.state["hotwords"]
                )
            except (TimeoutError, RequestError) as exc:
                self.events.add(
                    kind="notice",
                    text=f"認識できませんでした。もう一度話してください（{exc}）",
                )
                continue
            except Exception as exc:
                self._lost(link, str(exc))
                return
            finally:
                self.recognizing = False
            if text:
                self.events.add(
                    kind="heard",
                    text=text,
                    language=language,
                    latency_ms=round((time.time() - segment.speech_end_time) * 1000),
                    asr_ms=round((time.time() - started) * 1000),
                )

    def set_language(self, language: str) -> None:
        with self.lock:
            link = self.link
            if language not in LANGUAGE_NAMES:
                raise ValueError("言語を選んでください")
            if self.events.state["status"] != "ready":
                raise ValueError("接続してから言語を切り替えてください")
            if self.events.state["loading_language"]:
                raise ValueError("言語の切り替え中です")
            self.events.update(loading_language=language)
        threading.Thread(
            target=self._switch_language, args=(link, language), daemon=True
        ).start()

    def _switch_language(self, link: Link, language: str) -> None:
        try:
            self._load(link, language)
            self.events.update(
                language=language,
                loading_language="",
                **self._hotword_state(link, language),
            )
        except (TimeoutError, RequestError) as exc:
            self.events.update(loading_language="")
            self.events.add(
                kind="notice",
                text=f"{LANGUAGE_NAMES[language]}に切り替えられませんでした（{exc}）",
            )
        except Exception as exc:
            self.events.update(loading_language="")
            self._lost(link, str(exc))

    def set_hotwords(self, words: list[str]) -> None:
        self.events.update(hotwords=[word.strip() for word in words if word.strip()])

    # -- Speaking ----------------------------------------------------------

    def speak(self, text: str, voice: str = "") -> None:
        text = text.strip()
        if not text:
            raise ValueError("読み上げる文を入力してください")
        with self.lock:
            link = self.link
            if self.events.state["status"] != "ready" or link is None:
                raise ValueError("接続してから読み上げてください")
            if self.speaking.is_set():
                raise ValueError("読み上げが終わるまで待ってください")
            voice = voice or link.tts.default_voice
            if voice not in link.tts.voices:
                raise ValueError("声を選んでください")
            self.speaking.set()
        self.stop_speech.clear()
        threading.Thread(
            target=self._speak, args=(link, text, voice), daemon=True
        ).start()

    def _speak(self, link: Link, text: str, voice: str) -> None:
        sentences = split_sentences(text)
        item = self.events.add(
            kind="said",
            text=text,
            status="speaking",
            voice=voice,
            sentences=len(sentences),
            played=0,
        )
        self.events.update(activity="speaking")
        started = time.time()
        first = None
        try:
            for index, sentence in enumerate(sentences, start=1):
                for chunk in link.tts.synthesize(sentence, voice):
                    if first is None:
                        first = round((time.time() - started) * 1000)
                    if not link.speaker.play(
                        chunk, link.tts.sample_rate, self.stop_speech.is_set
                    ):
                        break
                if self.stop_speech.is_set():
                    break
                self.events.change(item, played=index, first_audio_ms=first)
            status = "stopped" if self.stop_speech.is_set() else "done"
            self.events.change(item, status=status, first_audio_ms=first)
        except (TimeoutError, RequestError) as exc:
            self.events.change(item, status="failed", error=str(exc))
        except Exception as exc:
            self.events.change(item, status="failed", error=str(exc))
            self._lost(link, str(exc))
        finally:
            self.muted_until = time.time() + MUTE_TAIL_S
            self.speaking.clear()
            if not self.options.listen:
                self.events.update(activity="")

    def stop(self) -> None:
        self.stop_speech.set()
