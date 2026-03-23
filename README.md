# qwen3-asr

Qwen3-ASR-1.7B を使った日本語音声文字起こし CLI ツール。
音声ファイル（wav / mp3 / mp4）を入力し、日本語テキストを出力します。

## 特徴

- 日本語特化の音声認識
- 対応フォーマット: wav / mp3 / mp4
- 長時間音声の自動チャンク分割（30秒単位）
- バックグラウンド知識による固有表現の認識精度向上
- GPU（CUDA）/ CPU 自動検出
- stdout 出力でパイプやリダイレクトと連携可能

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

### 基本的な文字起こし

```bash
qwen3-asr transcribe input.wav
```

mp3 / mp4 にも対応しています。

```bash
qwen3-asr transcribe meeting.mp3
qwen3-asr transcribe video.mp4
```

### 結果をファイルに保存

```bash
qwen3-asr transcribe input.wav -o result.txt
```

### バックグラウンド知識の指定

人名・用語・案件名等を記載したテキストファイルを `-c` オプションで指定すると、認識候補のバイアス付けにより固有表現の認識精度が向上します。

```bash
qwen3-asr transcribe meeting.mp3 -c context.txt
```

コンテキストファイルの例（`context.txt`）:

```
これは日本語の社内技術会議です。
出現しやすい語彙:
ナビゲーションズ, チームスピリット, ISO27001, Confluence,
Qwen3-ASR, AmiVoice, Whisper, vLLM
人名: 田中, 鈴木, 池田
案件名: 音声議事録PoC
```

効果的なコンテキストの書き方:

- 録音ごとに **10〜50語程度** の高頻度語彙と短い背景説明を記載する
- 人名、会社名、製品名、英字略語、カタカナ語が特に効果的
- 「句読点を減らす」「議事録風に直す」といった後処理指示は効果がない（認識候補のバイアス付けであり、指示追従ではないため）

### デバイスとモデルの指定

```bash
# CPU で実行（GPU がない環境向け）
qwen3-asr transcribe input.wav -d cpu

# 特定の GPU を指定
qwen3-asr transcribe input.wav -d cuda:0

# ローカルにダウンロード済みのモデルを指定
qwen3-asr transcribe input.wav -m /path/to/model
```

デフォルトでは CUDA GPU が利用可能なら自動的に使用し、なければ CPU にフォールバックします。

### 詳細ログの表示

```bash
qwen3-asr transcribe input.wav -v
```

デバイス情報、モデルロード状況、チャンク分割の進捗などが stderr に表示されます。

### オプション一覧

| オプション | 短縮形 | 説明 |
|---|---|---|
| `--output <path>` | `-o` | 結果をファイルに出力 |
| `--model <path>` | `-m` | モデルディレクトリの指定（デフォルト: `Qwen/Qwen3-ASR-1.7B`） |
| `--device <device>` | `-d` | デバイス指定（`cpu` / `cuda:0` など、デフォルト: 自動検出） |
| `--context <path>` | `-c` | バックグラウンド知識ファイル（人名・用語等のテキスト） |
| `--verbose` | `-v` | 詳細ログを表示 |
| `--version` | | バージョン表示 |
| `--help` | `-h` | ヘルプ表示 |

### パイプとの組み合わせ

文字起こし結果は stdout に出力されるため、他のコマンドと組み合わせて使えます。
ログメッセージは stderr に出力されるため、パイプやリダイレクトに影響しません。

```bash
# 結果をクリップボードにコピー（macOS）
qwen3-asr transcribe input.wav | pbcopy

# 結果の文字数をカウント
qwen3-asr transcribe input.wav | wc -m

# 結果を別のコマンドに渡す
qwen3-asr transcribe input.wav | grep "キーワード"
```

## 出力形式

文字起こし結果は句点（。）ごとに改行されます。

```
おはようございます。
本日の営業会議始めます。
総務グループからお願いいたします。
```

## 長時間音声の処理

30秒を超える音声ファイルは、自動的に30秒ごとのチャンクに分割して処理されます（2秒のオーバーラップあり）。`-v` オプションで進捗を確認できます。

```
$ qwen3-asr transcribe long_meeting.mp3 -v
デバイス: cuda:0, dtype: torch.bfloat16
モデルを読み込んでいます...
モデルの読み込みが完了しました。
文字起こし中: long_meeting.mp3
音声を 40 チャンクに分割しました（1104秒）
チャンク 1/40 を処理中...
チャンク 2/40 を処理中...
...
```

## 開発

```bash
# 開発用インストール
pip install -e ".[dev]"

# テスト実行
python -m pytest tests/ -v
```

## ライセンス

MIT
