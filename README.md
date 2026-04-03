---
title: picture-ai
emoji: 🖼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.10.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# picture-ai

人物サイズ比較ツール（Gradio）。

## 開発（ローカル起動）

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python pic.py
```

起動後、ブラウザでGradio UIが開きます。

## Git管理方針

- `dist/` `build/` `input/` `output/` はGit管理しません（`.gitignore` 済み）

## デプロイについて（重要）

最短で公開するなら **Hugging Face Spaces（Gradio）** が簡単です。

### Hugging Face Spaces（最短手順）

1. Hugging Face にログイン
2. New Space → SDK は **Gradio**
3. 既存のGitHubリポジトリ（`Yoichiro8552/picture-ai`）を接続
4. `app.py` がエントリーポイントになります（このリポジトリに追加済み）

補足：
- Spaces は `PORT=7860` が使われることが多いので `app.py` 側で対応しています

### Vercelについて

このプロジェクトは **Python + Gradio**（常駐プロセス）なので、Vercel（主にNext.js/静的/Serverless前提）とは相性が悪いです。
Vercelで公開したい場合は、Next.js化 + API分離などの構成変更が必要になります。

