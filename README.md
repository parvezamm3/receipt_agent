# レシートエージェント

OCR、データ抽出、検証、人間によるレビューを自動化するPythonベースのレシートPDF処理エージェントです。LangChain、Google Vertex AI、Gradioを活用し、堅牢な半自動ワークフローを実現します。

## 主な機能
- 新しいレシートPDFをフォルダ監視で自動検出
- PDFからレシート画像を抽出・切り出し
- Gemini/Vertex AIによるデータ抽出
- 抽出データの自動検証
- 失敗や不確実な場合はGradioによる人間レビュー
- 処理済みPDFの整理とデータログの自動管理

## はじめに

### 1. リポジトリのクローン
```bash
git clone https://github.com/parvezamm3/receipt_agent.git
cd ReceiptAgent
```

### 2. Conda環境のセットアップ
推奨: conda環境を作成・有効化
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 3. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定
プロジェクトルートに `.env` ファイルを作成し、Google Gemini APIキーを記載:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 5. フォルダ構成
以下のフォルダは自動生成されます:
- `pdfs/`（入力PDF）
- `images/`（切り出し画像）
- `output_pdfs/`（処理済みPDF）
- `success_pdfs/`（成功したPDF）
- `error_pdfs/`（人間レビュー用PDF）

### 6. エージェントの実行
```bash
python agent_controller.py
```
`pdfs/` フォルダを監視し、新しいPDFを自動処理します。

### 7. テストの実行
テストはダミーデータ・モックを利用し、実ファイルやAPIは使用しません。

全テスト実行:
```bash
python -m unittest test_agent_controller.py -v
python -m unittest test_tools.py -v
```

## 注意事項
- Google API利用時は `service_account_key.json` など必要な認証情報を配置してください。
- クローン手順のURLはご自身のリポジトリに合わせて修正してください。
- 本プロジェクトはWindows向けですが、他OSでも一部修正で動作可能です。

## ライセンス
MIT License 