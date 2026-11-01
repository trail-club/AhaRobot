"""aha_audio_tts: English TTS with Kokoro-82M.

Actions
  /aha/perception/audio/speak       aha_msgs/action/Speak   (cancel 可。複数 goal は順番に再生)
Topics
  /aha/perception/audio/tts_active  std_msgs/Bool           再生中 True (ASR はこの間マイクを無視)
  /aha/perception/audio/debug       std_msgs/String (JSON)  type=tts のイベント
  /aha/perception/audio/models      std_msgs/String (JSON)  読み込んだモデルの記録 (latched)
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import numpy as np
from aha_msgs.action import Speak
from rclpy.action import GoalResponse
from std_msgs.msg import Bool

from .audio_io import Speaker
from .common import ACTION_SPEAK, TOPIC_TTS_ACTIVE, AudioModelNode, spin_multithreaded
from .model_registry import Registry

SENTENCE_SPLIT = re.compile(r'(?<=[.!?;:])\s+|\n+')


@dataclass
class _PlayStats:
    ttfa: float | None = None         # goal 受理から最初の音まで
    synth: float = 0.0                # 合成にかかった合計
    audio: float = 0.0                # 再生した音声の長さ


class KokoroTts:
    def __init__(self, registry: Registry, device: str):
        from kokoro import KModel, KPipeline

        entry = registry.entry('tts', 'kokoro')
        spec = entry.spec
        with registry.loading(entry, device) as extra:
            local = entry.fetch(allow_patterns=['config.json', '*.pth', f'voices/{spec["lang_code"]}*.pt'])
            weights = next(local.glob('*.pth'))
            model = KModel(repo_id=entry.repo, config=str(local / 'config.json'),
                           model=str(weights)).to(device).eval()
            self.pipeline = KPipeline(lang_code=spec['lang_code'], repo_id=entry.repo, model=model)
            self.voices = {p.stem: str(p) for p in sorted((local / 'voices').glob('*.pt'))}
            self.default_voice = spec['voice']
            self.sample_rate = int(spec['sample_rate'])
            self.tag = entry.tag
            list(self.synthesize('Ready.', self.default_voice))  # warm up
            extra.update(weights=weights.name, voices=list(self.voices),
                         default_voice=self.default_voice)

    def synthesize(self, text: str, voice: str):
        """Yield float32 audio chunks for `text`."""
        for result in self.pipeline(text, voice=self.voices[voice]):
            if result.audio is not None:
                yield result.audio.detach().cpu().numpy().astype(np.float32)


class TtsNode(AudioModelNode):
    def __init__(self):
        super().__init__('aha_audio_tts')
        self.declare_parameter('output_device', '')
        self.declare_parameter('playback_rate', 24000)

        self.pub_active = self.create_publisher(Bool, TOPIC_TTS_ACTIVE, 10)

        self.get_logger().info('loading Kokoro ...')
        self.tts = KokoroTts(self.registry, self.device)
        self.publish_models()

        self.speaker = Speaker(self.get_parameter('output_device').value,
                               self.get_parameter('playback_rate').value)
        self._play_lock = threading.Lock()
        self.pub_active.publish(Bool(data=False))
        self._action = self.serve_action(Speak, ACTION_SPEAK, self._execute, self._on_goal)
        self.get_logger().info(f'ready: {self.tts.tag}, voice={self.tts.default_voice}')

    def _on_goal(self, goal) -> GoalResponse:
        if not goal.text.strip():
            return GoalResponse.REJECT
        if goal.voice and goal.voice not in self.tts.voices:
            self.get_logger().warn(f'reject speak goal: unknown voice "{goal.voice}"')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _play(self, sentences: list[str], voice: str, goal_handle, accepted: float,
              stats: _PlayStats) -> None:
        """文ごとに合成して再生する。キャンセルされたら途中で戻る."""
        cancelled = lambda: goal_handle.is_cancel_requested  # noqa: E731
        for i, sentence in enumerate(sentences):
            t0 = time.time()
            for chunk in self.tts.synthesize(sentence, voice):
                stats.synth += time.time() - t0
                if stats.ttfa is None:
                    stats.ttfa = time.time() - accepted
                stats.audio += len(chunk) / self.tts.sample_rate
                if not self.speaker.play(chunk, self.tts.sample_rate, cancelled):
                    return
                t0 = time.time()
            if cancelled():
                return
            goal_handle.publish_feedback(Speak.Feedback(progress=(i + 1) / len(sentences)))

    def _execute(self, goal_handle):
        text = goal_handle.request.text.strip()
        voice = goal_handle.request.voice or self.tts.default_voice
        result = Speak.Result()
        accepted = time.time()
        sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
        stats = _PlayStats()
        with self._play_lock:
            self.pub_active.publish(Bool(data=True))
            try:
                self._play(sentences, voice, goal_handle, accepted, stats)
            except Exception as e:
                self.get_logger().error(f'TTS failed: {e!r}')
                goal_handle.abort()
                result.message = repr(e)
                self._report(text, voice, stats, 'error')
                return result
            finally:
                self.pub_active.publish(Bool(data=False))

        completed = not goal_handle.is_cancel_requested
        result.time_to_first_audio = float(stats.ttfa or 0.0)
        status = 'done' if completed else 'canceled'
        self._report(text, voice, stats, status)
        if completed:
            goal_handle.succeed()
            result.success = True
        else:
            goal_handle.canceled()
        result.message = status
        return result

    def _report(self, text: str, voice: str, stats: _PlayStats, status: str) -> None:
        self.debug({
            'type': 'tts', 'text': text, 'voice': voice, 'model': self.tts.tag, 'status': status,
            'ttfa': round(stats.ttfa, 3) if stats.ttfa is not None else None,
            'synth_seconds': round(stats.synth, 3), 'audio_seconds': round(stats.audio, 2),
        })

    def destroy_node(self):
        try:
            self.speaker.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    spin_multithreaded(TtsNode, args)


if __name__ == '__main__':
    main()
