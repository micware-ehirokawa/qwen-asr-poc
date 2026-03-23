# 要件定義

## 1. 機能要件

### 1.1 音声文字起こし（transcribe コマンド）

- 音声ファイルのパスを引数として受け取り、文字起こし結果を標準出力に出力する
- 日本語固定で認識を行う（`language="Japanese"` を指定）

### 1.2 対応入力フォーマット

| フォーマット | 拡張子 | 備考 |
|---|---|---|
| WAV | `.wav` | PCM 非圧縮、最も基本的な形式 |
| MP3 | `.mp3` | 圧縮音声 |
| MP4 | `.mp4` | 動画ファイルから音声トラックを抽出して処理 |

`qwen-asr` SDK はローカルファイルパス・URL・base64・`(np.ndarray, sr)` 形式を直接受け付ける。本ツールではローカルファイルパスを入力とする。

### 1.3 出力

- 認識結果のテキストを標準出力に出力する
- デフォルト出力先: stdout（パイプやリダイレクトで利用可能）
- オプションで出力ファイルパスを指定可能（`-o` / `--output`）

## 2. CLI インターフェース

### 2.1 基本コマンド

```
qwen3-asr transcribe <audio_file> [options]
```

### 2.2 オプション

| オプション | 短縮形 | 説明 |
|---|---|---|
| `--output <path>` | `-o` | 結果をファイルに出力 |
| `--model <path>` | `-m` | モデルディレクトリの指定（デフォルト: `Qwen/Qwen3-ASR-1.7B` を自動ダウンロード） |
| `--device <device>` | `-d` | `device_map` の指定（`cpu` / `cuda:0` など、デフォルト: 自動検出） |
| `--verbose` | `-v` | 詳細ログを表示 |
| `--version` | | バージョン表示 |
| `--help` | `-h` | ヘルプ表示 |

## 3. 非機能要件

### 3.1 モデル管理

- `Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B")` によりモデル名指定で Hugging Face Hub から自動ダウンロードされる
- ダウンロード済みモデルは Hugging Face のローカルキャッシュ（`~/.cache/huggingface/`）に保存され、2回目以降は再ダウンロード不要
- `--model` オプションで事前ダウンロード済みのローカルパスを指定可能（オフライン環境対応）

### 3.2 デバイス対応

- CPU での推論をサポート（必須）
- CUDA GPU が利用可能な場合は自動的に GPU を使用（`device_map="cuda:0"`）
- GPU 推論時の dtype は `torch.bfloat16` を使用（公式推奨）
- CPU 実行時は `torch.float32` を使用（bfloat16 は AMD CPU で最大1000倍の性能劣化、ハング報告あり）

### 3.3 リソース要件

| 環境 | メモリ要件 |
|---|---|
| GPU VRAM（BF16） | 約 3.9 GB |
| システム RAM（推奨） | 16 GB 以上 |

### 3.4 音声前処理

- `qwen-asr` SDK の `transcribe()` がローカルファイルパスを直接受け付けるため、SDK に処理を委譲する
- SDK 内部では `librosa` / `soundfile` が音声処理を担当する
- mp3 / mp4 のデコードにはシステムに ffmpeg が必要（SDK の依存である librosa が利用する）

### 3.5 エラーハンドリング

- ファイルが存在しない場合: 明確なエラーメッセージを表示して終了
- 非対応フォーマットの場合: 対応フォーマット一覧を提示して終了
- モデルダウンロード失敗時: リトライ案内を表示して終了
- ffmpeg 未インストール時（mp3/mp4 使用時）: インストール案内を表示

## 4. パッケージ要件

### 4.1 配布形式

- pyproject.toml ベースのパッケージ構成
- `pip install` でインストール可能
- インストール後 `qwen3-asr` コマンドとして実行可能（エントリポイント登録）

### 4.2 Python バージョン

- Python 3.9 以上（`qwen-asr` パッケージの要件）
- 推奨: Python 3.12（公式セットアップ例で使用）

### 4.3 依存ライブラリ

| ライブラリ | 用途 | 備考 |
|---|---|---|
| `torch` | テンソル演算・推論エンジン | `qwen-asr` の依存に含まれず別途インストール必要 |
| `qwen-asr` | Qwen3-ASR モデルのロード・推論 | transformers, accelerate, librosa, soundfile 等を内包 |
| `click` | CLI フレームワーク | 依存ゼロ、8.3.x で安定。typer は pre-1.0 で依存が多いため不採用 |

> **注意**: `qwen-asr` は `gradio`, `flask` 等の Web 系ライブラリも依存に含む。CLI 専用ツールとしては過剰だが、公式 SDK であるためそのまま利用する。

#### 外部依存

| ツール | 用途 | 備考 |
|---|---|---|
| `ffmpeg` | mp3 / mp4 のデコード | システムに別途インストールが必要。WAV のみの場合は不要 |

#### オプション依存（高速化）

| ライブラリ | 用途 |
|---|---|
| `qwen-asr[vllm]` | vLLM バックエンドによる高速推論 |
| `flash-attn` | FlashAttention 2 による GPU メモリ効率向上（BF16/FP16 前提） |

## 5. 内部設計メモ

### 5.1 モデル呼び出しの基本パターン

```python
import torch
from qwen_asr import Qwen3ASRModel

# GPU 利用時
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# CPU 利用時
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.float32,
    device_map="cpu",
)

results = model.transcribe(
    audio="input.wav",
    language="Japanese",
)

print(results[0].text)
```

### 5.2 デバイス自動検出ロジック

```
CUDA 利用可能 → device_map="cuda:0", dtype=torch.bfloat16
CUDA 利用不可 → device_map="cpu", dtype=torch.float32
--device 指定あり → ユーザー指定値を優先
```

## 6. スコープ外

- Web UI / GUI
- REST API サーバー
- リアルタイムストリーミング認識
- 日本語以外の言語対応
- モデルのファインチューニング機能

## 7. 将来拡張候補

- バッチ処理（複数ファイル同時指定）
- タイムスタンプ付き出力（`Qwen3-ForcedAligner-0.6B` との連携、`return_time_stamps=True`）
- vLLM バックエンドによるサーバー運用
