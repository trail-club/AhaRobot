# 配線資料

配線ごとに `docs/electrical/<対象名>/` を作り、WireViz YAML・WireViz出力SVG・機器配置SVG・元資料をまとめて管理する。
各ディレクトリの `README.md` には、その回路図固有の補足説明を記載する。

## 資料一覧

- [昇降機構（lift）](lift/)

## 追加・更新方法

1. 対象のディレクトリに元資料と配線を記述したWireViz YAMLを置く。
2. Python 3.11・uv・Graphviz（`dot`）を用意し、YAMLからSVGを生成する。リポジトリ直下での実行例:

   ```bash
   uvx --python 3.11 --from wireviz==0.4.1 wireviz -f s docs/electrical/lift/aha_lift_wireviz.yml
   ```

3. YAMLと[機器配置図の生成プロンプト](layout_prompt.md)をAIに渡して機器配置SVGを作る。
4. 全接続・線色・未接続線・内部接続をYAMLと照合し、資料一式をコミットする。新規追加時は上の一覧に対象を追記する。
