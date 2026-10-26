# 昇降機構の配線資料

[Slackスレッド](https://trail-weblab.slack.com/archives/C0C6W9H055M/p1792990590106589)の添付4点をそのまま収録。実機との照合は未実施。

| 資料 | 内容 |
| --- | --- |
| [WireViz YAML](aha_lift_wireviz.yml) | 配線の編集元 |
| [WireViz出力](aha_lift_wireviz.svg) | 端子間の接続図 |
| [機器配置図](aha_lift_layout.svg) | 機器と配線を見渡す図 |
| [元PDF](aharobot_lift_schematic.pdf) | AHA-LIFT-SCH-001 REV 1.0（2026-10-25） |

## 更新手順

1. YAMLの接続情報を編集する。
2. Python 3.11・uv・Graphviz（`dot`）を用意し、リポジトリ直下でWireViz出力を更新する。

   ```bash
   uvx --python 3.11 --from wireviz==0.4.1 wireviz -f s docs/electrical/lift/aha_lift_wireviz.yml
   ```

3. YAMLと[スレッドで使用したプロンプト](layout_prompt.md)をAIに渡して機器配置図を更新し、全接続・線色・未接続線・内部接続をYAMLと照合する。
