# submoduleの変更・更新

`upstream/` は特定commitを参照する。Astra系6件は `trail-club` のfork、
SOBITS / TMC資源はTeamSOBITSを直接参照する。URLは [.gitmodules](../.gitmodules)を参照。

## コードを変更する

submodule内で作業ブランチを作り、fork側でコミット・PRを作る。
AhaRobot側で記録するのはファイルの差分ではなく、使用するcommitのSHA。

```bash
cd upstream/astra_description
git remote -v  # originが変更先のforkであることを確認
git switch -c fix-xacro
# 編集・検証
git add <変更したファイル>
git commit -m "Fix robot description"
git push -u origin fix-xacro
```

fork側のPRをマージしたら、そのcommitを参照する。

```bash
git fetch origin
git switch --detach <マージ後のSHA>
cd ../..
make test
git add upstream/astra_description
git commit -m "Update astra_description reference"
```

AhaRobot側でもSHA更新のPRを作る。Squash mergeの場合は作業ブランチとマージ後のSHAが異なる。

## 元の開発元の更新を取り込む

Astra系では、作業するcloneで `upstream` remoteを登録する。設定は別cloneへ共有されない。

```bash
cd upstream/astra_description
git remote add upstream https://github.com/hilookas/astra_description.git  # 未登録の場合
git fetch upstream
git switch -c sync-upstream origin/main
git merge upstream/main
# 競合解消・検証後、originへpushしてfork側のPRを作る
```

fork側のマージ後は上記のSHA更新手順を使う。
通常の取得は [開発コンテナの起動フロー](docker.md#起動フロー)を参照。
