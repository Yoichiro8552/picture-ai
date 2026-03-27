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

このプロジェクトは **Python + Gradio** です。Vercelは主に Next.js などのNode/静的サイト向けなので、
GradioアプリをそのままVercelに載せるのは相性が悪い/制限に当たりやすいです。

おすすめは以下です：

- Hugging Face Spaces（Gradioをそのまま公開しやすい）
- Render / Railway / Fly.io / Cloud Run（Dockerで常駐プロセスとして動かす）

Vercelにどうしても載せたい場合は、構成変更（Next.jsフロント + API分離 等）が必要になります。

