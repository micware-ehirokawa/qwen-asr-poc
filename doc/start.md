# qwen3-asr プロジェクト概要

## 目的

Qwen3-ASR-1.7B をローカル環境で動作させ、日本語音声の文字起こしを行う CLI ツールを開発する。

## 使用モデル

- **Qwen3-ASR-1.7B** (Alibaba Cloud / Qwen チーム)
- 音声認識(ASR: Automatic Speech Recognition)に特化した軽量モデル

## 機能要件

- 音声ファイルを入力し、日本語テキストを出力する
- 対応フォーマット: wav, mp3, mp4
- 日本語特化の文字起こし
- CLI（コマンドラインインターフェース）のみ

## 技術スタック

- Python
- 頒布可能なパッケージ（pip install 可能な形式）として構成

## スコープ

### 今回やること

- CLI ツールの実装
- ローカルでのモデル推論
- パッケージとしての構成（pyproject.toml / setup）

### 今回やらないこと

- Web UI / GUI
- API サーバー
- リアルタイムストリーミング認識
- 多言語対応

## 想定される使い方

```bash
# 音声ファイルを指定して文字起こし
qwen3-asr transcribe input.wav

# mp3 / mp4 にも対応
qwen3-asr transcribe meeting.mp3
qwen3-asr transcribe video.mp4
```
