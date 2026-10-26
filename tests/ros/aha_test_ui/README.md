# aha_test_ui

ロボットの状態を人が見るためのモニタ。各モジュールが今の状態・LLMとの会話・イベントをJSON（`std_msgs/String`）で出し、
`robot_monitor.py`（Qtのウィンドウ）がまとめて表示する。

```bash
ros2 run aha_test_ui robot_monitor.py
```

ウィンドウは上から次の3つ。

- 状態：発信元ごとに1行（`source` / `state` / `detail` / 更新からの時間）
- 会話：LLMとの会話を吹き出しで表示。結果（置いた・失敗した理由など）は該当する回答の後に注記。コンボボックスで起動後の会話を選ぶ（新しいものが上、始まると自動で選択）。`system` でシステムプロンプトも表示
- イベント：時刻・発信元・内容のログ（警告は黄、エラーは赤、最新500行）

## topic

| topic | 内容 | QoS |
| --- | --- | --- |
| `/aha/ui/status` | `{"source", "state", "detail", "stamp"}`：発信元の今の状態 | transient local（発信元ごとに最新1件） |
| `/aha/ui/chat` | `{"source", "id", "title", "messages", "notes", "done"}`：会話（`messages` はchat completionsのメッセージ、`notes` はメッセージの番号→`{"text", "level"}` の注記）。同じ `source` と `id` の記録で置き換える | transient local（10件） |
| `/aha/ui/events` | `{"source", "level", "text", "stamp"}`：1件のログ（`level` は `info` / `warn` / `error`） | volatile |

`stamp` はwall clockの秒。形式と描画は `aha_test_ui.records`、発行は `aha_test_ui.publisher.UiPublisher` を使う。

```python
from aha_test_ui.publisher import UiPublisher
from aha_test_ui.records import note

ui = UiPublisher(node, "my_module")
ui.status("searching", "table 1")
ui.event("object not found", "warn")
ui.chat("apple-1", "apple", messages, notes={2: [note("placed", "ok")]}, done=True)
```

今の発信元は `place_runner`（[aha_test_orchestration](../aha_test_orchestration/README.md)：観測・判断・置く・台車の移動、置く場所の会話）と
`place_server`（[aha_test_manipulation](../aha_test_manipulation/README.md#置く動作)：置く動作の段階と候補の番号、結果）。
