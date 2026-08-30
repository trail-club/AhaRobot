# Upstream 追従ワークフロー

`upstream/*` の各 submodule は `trail-club/<name>` の fork を指す。
上流は `hilookas/<name>`。fork 側で `upstream` remote を設定し、明示的に取り込む。

## 初回セットアップ (fork ローカルクローンで一度だけ)

```bash
# 例: astra_description
git clone git@github.com:trail-club/astra_description.git
cd astra_description
git remote add upstream https://github.com/hilookas/astra_description.git
git fetch upstream
```

対応表:

| fork (origin) | upstream |
| --- | --- |
| trail-club/astra_description | hilookas/astra_description |
| trail-club/astra_controller | hilookas/astra_controller |
| trail-club/astra_controller_interfaces | hilookas/astra_controller_interfaces |
| trail-club/astra_moveit_config | hilookas/astra_moveit_config |
| trail-club/AstraFirmwares | hilookas/AstraFirmwares |
| trail-club/Astra_Hardwares | hilookas/Astra_Hardwares |

## 上流の変更を取り込む

```bash
cd upstream/<name>            # super repo 内の submodule で作業
git fetch upstream
git checkout main
git merge upstream/main       # or: git rebase upstream/main
# 競合解消 → push
git push origin main
cd ../..
git add upstream/<name>       # super repo 側で SHA を進める
git commit -m "chore(upstream): bump <name> to <short-sha>"
```

## チーム改変ブランチ

改変は fork 側で feature branch を切って行う (`main` に直接 push しない):

```bash
cd upstream/<name>
git checkout -b jazzy-port     # 例
# 編集 → commit → push
git push -u origin jazzy-port
```

汎用的な修正 (バグ, Jazzy 対応など) は upstream (`hilookas/<name>`) に PR を出す。
本チーム固有の設定 (競技用 world / launch / params) は upstream に戻さず overlay workspace 側に置く。

## Super repo 側の submodule 操作

```bash
# clone 直後
git submodule update --init --recursive

# 全 submodule を各 tracking branch 最新へ
git submodule update --remote --merge

# 特定 submodule だけ最新に
git -C upstream/astra_controller pull origin main
git add upstream/astra_controller
git commit -m "chore(upstream): bump astra_controller"
```
