"""aha_audio_debug_ui: ブラウザで音声スタックの状態をリアルタイムに見るための Web UI.

http://localhost:8765 を開く。/aha/perception/audio/{debug,models,tts_active} を WebSocket で
中継し、UI からの操作 (言語切り替え・ホットワード・TTS テスト・Listen テスト) を
parameter / action として各ノードへ送る。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from importlib import resources

import rclpy
from aha_msgs.action import Listen, Speak
from aiohttp import WSMsgType, web
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from std_msgs.msg import Bool, String

from .common import (ACTION_LISTEN, ACTION_SPEAK, SEGMENT_NAME, SEGMENTS_DIR, TOPIC_DEBUG,
                     TOPIC_MODELS, TOPIC_TTS_ACTIVE, json_text, latched_qos, notice_json)


class DebugUiNode(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__('aha_audio_debug_ui')
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 8765)
        self.declare_parameter('segments_dir', SEGMENTS_DIR)
        self.declare_parameter('asr_node', 'aha_audio_asr')

        self.loop = loop
        self.ws_clients: set[web.WebSocketResponse] = set()
        self.models: dict[str, dict] = {}
        self.last_asr_state: str | None = None
        self.tts_active = False
        self._speak_goal = None

        self.create_subscription(String, TOPIC_DEBUG, self._on_debug, 100)
        # 複数ノード (asr / tts) がそれぞれ latched で出すので、受信側は全員分を保持できる深さにする
        self.create_subscription(String, TOPIC_MODELS, self._on_models, latched_qos(10))
        self.create_subscription(Bool, TOPIC_TTS_ACTIVE, self._on_tts_active, 10)
        self.asr_params = AsyncParameterClient(self, self.get_parameter('asr_node').value)
        self.speak_client = ActionClient(self, Speak, ACTION_SPEAK)
        self.listen_client = ActionClient(self, Listen, ACTION_LISTEN)

    # ------------------------------------------------------------- ROS -> UI
    def _on_debug(self, msg: String) -> None:
        payload = json.loads(msg.data)
        if payload.get('type') == 'asr_state':
            self.last_asr_state = msg.data
        self.broadcast(msg.data)

    def _on_models(self, msg: String) -> None:
        record = json.loads(msg.data)
        self.models[record['node']] = record
        self.broadcast(self._models_json())

    def _on_tts_active(self, msg: Bool) -> None:
        self.tts_active = msg.data
        self.broadcast(self._tts_active_json())

    def _models_json(self) -> str:
        return json_text({'type': 'models', 'models': self.models})

    def _tts_active_json(self) -> str:
        return json_text({'type': 'tts_active', 'active': self.tts_active})

    def broadcast(self, text: str) -> None:
        self.loop.call_soon_threadsafe(self._broadcast, text)

    def _broadcast(self, text: str) -> None:
        for ws in list(self.ws_clients):
            if ws.closed:
                self.ws_clients.discard(ws)
            else:
                asyncio.ensure_future(ws.send_str(text))

    def notice(self, level: str, text: str) -> None:
        self.broadcast(notice_json(level, text))

    def snapshot(self) -> list[str]:
        out = [self._models_json(), self._tts_active_json()]
        if self.last_asr_state:
            out.append(self.last_asr_state)
        return out

    # ------------------------------------------------------------- UI -> ROS
    def handle_command(self, cmd: dict) -> None:
        kind = cmd.get('cmd')
        if kind == 'set_language':
            self._set_asr_param(Parameter('language', value=str(cmd['language'])))
        elif kind == 'set_hotwords':
            words = [str(w).strip() for w in cmd.get('hotwords', []) if str(w).strip()]
            self._set_asr_param(Parameter('hotwords', Parameter.Type.STRING_ARRAY, words))
        elif kind == 'speak':
            self._send_speak(str(cmd.get('text', '')), str(cmd.get('voice', '')))
        elif kind == 'cancel_speak':
            if self._speak_goal is not None:
                self._speak_goal.cancel_goal_async()
        elif kind == 'listen':
            self._send_listen(str(cmd.get('language', '')), float(cmd.get('timeout', 0.0)))
        else:
            self.notice('error', f'unknown command: {kind}')

    def _set_asr_param(self, param: Parameter) -> None:
        if not self.asr_params.services_are_ready():
            self.notice('error', 'aha_audio_asr が見つかりません')
            return
        fut = self.asr_params.set_parameters([param])

        def done(f):
            r = f.result().results[0]
            if r.successful:
                self.notice('info', f'{param.name} を更新しました')
            else:
                self.notice('error', f'{param.name} の更新に失敗: {r.reason}')
        fut.add_done_callback(done)

    def _send_speak(self, text: str, voice: str) -> None:
        if not text.strip():
            self.notice('warn', '話すテキストが空です')
            return
        if not self.speak_client.server_is_ready():
            self.notice('error', 'aha_audio_tts が見つかりません')
            return
        fut = self.speak_client.send_goal_async(Speak.Goal(text=text, voice=voice))

        def accepted(f):
            handle = f.result()
            if not handle.accepted:
                self.notice('error', 'TTS goal が拒否されました')
                return
            self._speak_goal = handle
        fut.add_done_callback(accepted)

    def _send_listen(self, language: str, timeout: float) -> None:
        if not self.listen_client.server_is_ready():
            self.notice('error', 'aha_audio_asr の listen action が見つかりません')
            return
        fut = self.listen_client.send_goal_async(Listen.Goal(language=language, timeout=timeout))

        def accepted(f):
            handle = f.result()
            if not handle.accepted:
                self.notice('error', 'Listen goal が拒否されました (実行中の Listen がある?)')
                return
            self.notice('info', 'Listen: 発話を待っています…')
            handle.get_result_async().add_done_callback(finished)

        def finished(f):
            res = f.result().result
            if res.success:
                self.notice('info', f'Listen 結果: "{res.transcript.text}"')
            else:
                self.notice('warn', f'Listen 失敗: {res.message}')
        fut.add_done_callback(accepted)


def build_app(node: DebugUiNode) -> web.Application:
    index_html = resources.files(__package__).joinpath('web/index.html').read_text('utf-8')
    segments_dir = node.get_parameter('segments_dir').value

    async def index(_):
        return web.Response(text=index_html, content_type='text/html')

    async def segment(request):
        name = request.match_info['name']
        path = os.path.join(segments_dir, name)
        if not SEGMENT_NAME.match(name) or not os.path.isfile(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Cache-Control': 'no-store'})

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        node.ws_clients.add(ws)
        for text in node.snapshot():
            await ws.send_str(text)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    node.handle_command(json.loads(msg.data))
                except Exception as e:
                    await ws.send_str(notice_json('error', repr(e)))
        node.ws_clients.discard(ws)
        return ws

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/segments/{name}', segment)
    return app


def main(args=None):
    rclpy.init(args=args)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    node = DebugUiNode(loop)

    def spin():
        # rclpy は SIGINT で context を shutdown するだけなので、spin が抜けたら web 側も止める
        try:
            rclpy.spin(node)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
    threading.Thread(target=spin, daemon=True).start()

    host = node.get_parameter('host').value
    port = node.get_parameter('port').value
    runner = web.AppRunner(build_app(node))
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, host, port).start())
    node.get_logger().info(f'debug UI: http://{"localhost" if host in ("0.0.0.0", "127.0.0.1") else host}:{port}')
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(runner.cleanup())
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
