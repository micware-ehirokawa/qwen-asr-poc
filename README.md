# qwen3-asr

Qwen3-ASR-1.7B を使った日本語音声文字起こし CLI ツール

## 前提条件

- Python 3.9 以上（推奨: 3.12）
- ffmpeg（mp3 / mp4 を使用する場合）

## インストール

```bash
pip install .
```

## 使い方

```bash
# 音声ファイルを文字起こし
qwen3-asr transcribe input.wav

# mp3 / mp4 にも対応
qwen3-asr transcribe meeting.mp3
qwen3-asr transcribe video.mp4

# 結果をファイルに保存
qwen3-asr transcribe input.wav -o result.txt

# モデルパスやデバイスを指定
qwen3-asr transcribe input.wav -m /path/to/model -d cpu

# 詳細ログ
qwen3-asr transcribe input.wav -v
```

初回実行時にモデル（約 3.9 GB）が Hugging Face Hub から自動ダウンロードされます。

## 開発

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```
