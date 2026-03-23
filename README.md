# qwen3-asr

Qwen3-ASR-1.7B を使った日本語音声文字起こし CLI ツール。
音声ファイル（wav / mp3 / mp4）を入力し、日本語テキストを出力します。

## 前提条件

- Python 3.9 以上（推奨: 3.12）
- CUDA 対応 GPU（推奨、VRAM 約 3.9 GB 以上）または CPU（動作するが低速）
- システム RAM 16 GB 以上（推奨）
- ffmpeg（mp3 / mp4 を使用する場合）

## インストール

### 1. Python 環境の準備

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. パッケージのインストール

```bash
pip install .
```

初回実行時にモデル（約 3.9 GB）が Hugging Face Hub から `~/.cache/huggingface/` に自動ダウンロードされます。

### 3. ffmpeg のインストール（mp3 / mp4 を使う場合）

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

WAV ファイルのみ使用する場合、ffmpeg は不要です。

### 4. オフライン環境でのモデル準備（任意）

ネットワークに接続できない環境では、事前にモデルをダウンロードしておきます。

```bash
# オンライン環境でダウンロード
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir /path/to/model

# オフライン環境で使用
qwen3-asr transcribe input.wav -m /path/to/model
```

## 使い方

```bash
# 基本的な文字起こし
qwen3-asr transcribe input.wav

# mp3 / mp4 にも対応
qwen3-asr transcribe meeting.mp3
qwen3-asr transcribe video.mp4

# 結果をファイルに保存
qwen3-asr transcribe input.wav -o result.txt

# デバイスを明示指定（デフォルトは自動検出）
qwen3-asr transcribe input.wav -d cpu
qwen3-asr transcribe input.wav -d cuda:0

# ローカルのモデルを指定
qwen3-asr transcribe input.wav -m /path/to/model

# 詳細ログを表示
qwen3-asr transcribe input.wav -v
```

長時間の音声ファイルは自動的に 30 秒ごとのチャンクに分割して処理されます。

### オプション一覧

| オプション | 短縮形 | 説明 |
|---|---|---|
| `--output <path>` | `-o` | 結果をファイルに出力 |
| `--model <path>` | `-m` | モデルディレクトリの指定（デフォルト: `Qwen/Qwen3-ASR-1.7B`） |
| `--device <device>` | `-d` | デバイス指定（`cpu` / `cuda:0` など、デフォルト: 自動検出） |
| `--verbose` | `-v` | 詳細ログを表示 |
| `--version` | | バージョン表示 |
| `--help` | `-h` | ヘルプ表示 |

### パイプとの組み合わせ

文字起こし結果は stdout に出力されるため、他のコマンドと組み合わせて使えます。

```bash
# 結果をクリップボードにコピー（macOS）
qwen3-asr transcribe input.wav | pbcopy

# 結果の文字数をカウント
qwen3-asr transcribe input.wav | wc -m

# 結果を別のコマンドに渡す
qwen3-asr transcribe input.wav | grep "キーワード"
```

ログメッセージは stderr に出力されるため、パイプやリダイレクトに影響しません。

## 開発

```bash
# 開発用インストール
pip install -e ".[dev]"

# テスト実行
python -m pytest tests/ -v
```

## ライセンス

MIT
