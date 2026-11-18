# GPSRの計画の評価

[aha_tasks](../../../overlay_ws/src/aha_tasks/README.md#命令の聞き取りと計画)の計画器（`aha_tasks.gpsr_planner`）を、公式のコマンド生成器の命令でROSなしに評価する。

1. 生成器（[RoboCupAtHome/CommandGenerator](https://github.com/RoboCupAtHome/CommandGenerator)。コードは取り込まず、`--generator` のcloneをimportする）に、`config/gpsr.json` の `gpsr.names` と `config/vocabulary/gpsr.json` の部屋・地点・物体を生成器のデータ（CompetitionTemplateの形）として渡し、命令を作る。種類はpeople、objects、どちらでも、の順に繰り返す。
2. 文法の解析（`aha_tasks.gpsr_grammar`）の手順を正解とし、LLMの計画と比べる。人の句と挨拶の文は比べない。
3. 先頭の `--rephrase` 件は、生成器のGPSR UIと同じ言い換えのプロンプト（`llm.py`）で同じLLMに3つずつ言い換えさせ、元の命令の正解と比べる。
4. 生成順に3命令ずつ並べ替え、手順数、枠（20）に入るか、命令を入れ替えたか、直線距離の削減を出す。

## 実行

DGXのLLMに届くPCで、リポジトリのルートから実行する（ROS不要。PyYAML・numpy・Pillow・requestsを使う）。

```bash
git clone https://github.com/RoboCupAtHome/CommandGenerator /tmp/CommandGenerator
uv run --no-project --with pyyaml --with numpy --with pillow --with requests \
  python tests/orchestration/gpsr_plan/eval_plans.py --generator /tmp/CommandGenerator \
  --count 100 --rephrase 20 --output tests/orchestration/gpsr_plan/results/run.json
```

| オプション | 既定 | 内容 |
| --- | --- | --- |
| `--generator` | 必須 | CommandGeneratorのclone |
| `--count` / `--seed` | `60` / `1` | 生成する命令の数と乱数のseed |
| `--rephrase` | `10` | 言い換える命令の数（3つずつ） |
| `--task` | `gpsr` | タスク定義と語彙の名前 |
| `--llm-url` / `--model` | DGXのQwen | LLM |
| `--thinking` / `--attempts` / `--timeout` | なし / `3` / `90` | 計画器の思考、LLMの試行、命令ごとの秒 |
| `--output` | なし | 命令・計画・誤りのJSON。`results/` はGit対象外 |

出力は、文法で読めた命令の数、LLMの有効な計画と正解と同じ計画の割合・時間（生成のまま / 言い換え）、正解と違った命令、3命令の並べ替えの要約。
