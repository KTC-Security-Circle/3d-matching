生成AIを使用し作成しています

# 3d-matchin

3d-matchingは、Python 3.12環境で動作するサンプルプロジェクトです。

## 使用技術

- Python3.12
- uv

## 構成

```
/app
├── pyproject.toml
├── pyrightconfig.json
├── README.md
├── uv.lock
├── src/
├── .devcontainer/
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── .ruff.toml
├── .venv/
└── .vscode/
```

## セットアップ

### 1. リポジトリのクローン

```bash
git clone git@github.com:haruki26/3d-matching.git
```

### 2. 開発コンテナの起動

VS CodeのDev Containers機能で開発用コンテナを起動します。初回起動時はビルドの関係で遅くなります。

自動的に`uv sync`が実行されて依存関係の追加が完了します。

#### 開発コンテナからのプッシュ

以下記事を参考にホストマシンで設定を行うことでsshを使用したプッシュが可能となります

- https://zenn.dev/nayushi/articles/5d577c93e03a9b#wsl2%E3%81%AEssh-agent%E3%81%AF%E4%B8%80%E5%91%B3%E9%81%95%E3%81%86%E3%82%88%E3%81%86%E3%81%A0


## Lint・型チェック

- Lint: `uv run ruff check ./src`
- フォーマット: `uv run ruff check ./src --fix`
- 型チェック: `uv run pyright`

## その他

pre-commitを設定してあるためコミット前にチェックが行われます。
