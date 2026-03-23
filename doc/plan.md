# qwen3-asr 実装計画

## 1. プロジェクト構成（ディレクトリレイアウト）

```
qwen3-asr/
├── CLAUDE.md
├── LICENSE
├── README.md
├── pyproject.toml
├── .gitignore
├── doc/
│   ├── start.md
│   ├── requirements.md
│   ├── TDR_qwen-asr-python.md
│   └── plan.md
├── src/
│   └── qwen3_asr/
│       ├── __init__.py           # バージョン定義
│       ├── cli.py                # Click ベースの CLI
│       ├── transcriber.py        # モデルロード・推論ロジック
│       └── exceptions.py         # カスタム例外クラス
└── tests/
    ├── __init__.py
    ├── conftest.py               # pytest フィクスチャ（モック等）
    ├── test_cli.py               # CLI のテスト
    └── test_transcriber.py       # 推論ロジックのテスト
```

### 設計方針

- **src レイアウト** を採用。未インストールのパッケージを誤ってインポートする事故を防ぐ。
- **関心の分離**: CLI 層（`cli.py`）とモデル推論層（`transcriber.py`）を明確に分離する。
- **例外クラス**: `exceptions.py` に集約し、CLI 層でキャッチして日本語メッセージに変換する。

---

## 2. 実装フェーズ

### フェーズ 1: プロジェクト基盤（パッケージング・エントリポイント）

**目的**: `pip install -e .` でインストールでき、`qwen3-asr --help` が動作する最小構成を作る。

#### 1-1. `pyproject.toml` の作成

| 項目 | 値 |
|---|---|
| ビルドシステム | `hatchling`（PEP 621 準拠、設定のみで完結） |
| パッケージ名 | `qwen3-asr` |
| バージョン | `0.1.0`（`src/qwen3_asr/__init__.py` から動的取得） |
| Python 要件 | `>=3.9` |
| 依存ライブラリ | `torch>=2.0`, `qwen-asr`, `click>=8.0,<9` |
| エントリポイント | `qwen3-asr = qwen3_asr.cli:main` |
| 開発用依存 | `pytest`, `pytest-mock` |

#### 1-2. `src/qwen3_asr/__init__.py` の作成

- `__version__ = "0.1.0"` を定義する。

#### 1-3. `src/qwen3_asr/cli.py` のスケルトン作成

- `click.group()` で `main` グループを定義する。
- `--version` オプションを `click.version_option()` で登録する。
- `transcribe` サブコマンドのスケルトンを定義する。

#### 1-4. 動作確認

```bash
pip install -e ".[dev]"
qwen3-asr --version
qwen3-asr --help
qwen3-asr transcribe --help
```

---

### フェーズ 2: 例外クラスとバリデーション

**目的**: エラーハンドリングの基盤を先に整備する。

#### 2-1. `src/qwen3_asr/exceptions.py` の作成

| 例外クラス | 発生条件 |
|---|---|
| `Qwen3AsrError` | 共通基底クラス |
| `AudioFileNotFoundError` | 指定された音声ファイルが存在しない |
| `UnsupportedFormatError` | 対応外の拡張子（`.wav`, `.mp3`, `.mp4` 以外） |
| `ModelLoadError` | モデルのダウンロードまたはロードに失敗 |
| `FfmpegNotFoundError` | mp3/mp4 処理時に ffmpeg が見つからない |

#### 2-2. `cli.py` へのバリデーション追加

1. **ファイル存在チェック**: `pathlib.Path.exists()`
2. **拡張子チェック**: `pathlib.Path.suffix.lower()`
3. **ffmpeg チェック**: `.mp3` / `.mp4` の場合、`shutil.which("ffmpeg")` で確認

`main` グループに `try/except Qwen3AsrError` のラッパーを設け、stderr にメッセージ出力して `sys.exit(1)` する。

---

### フェーズ 3: モデル推論ロジック（コア機能）

**目的**: `transcriber.py` にモデルのロードと推論を実装する。

#### 3-1. `src/qwen3_asr/transcriber.py` の作成

```
関数: detect_device(device: str | None) -> tuple[str, torch.dtype]
  - CUDA 利用可 → ("cuda:0", torch.bfloat16)
  - CPU → ("cpu", torch.float32)
  - 明示指定があればそれを優先

関数: load_model(model_path: str, device_map: str, dtype: torch.dtype) -> Qwen3ASRModel
  - from_pretrained() を呼び出す
  - 失敗時は ModelLoadError を送出

関数: transcribe(model: Qwen3ASRModel, audio_path: str) -> str
  - model.transcribe(audio=audio_path, language="Japanese")
  - results[0].text を返す
```

**設計判断**:
- モデルをクラスでラップしない。ワンショット実行のため関数群で十分。
- `language="Japanese"` はハードコード（要件で日本語固定）。

#### 3-2. `cli.py` の `transcribe` コマンドを完成させる

処理フロー:

```
1. バリデーション（フェーズ 2 で実装済み）
2. detect_device(device) でデバイスと dtype を決定
3. load_model(model_path, device_map, dtype) でモデルをロード
4. transcribe(model, audio_path) で文字起こし実行
5. 結果を出力（--output 指定時はファイル、なければ stdout）
```

**Click オプション定義**:

- `audio_file`: `click.argument` — `click.Path(exists=False)` を使用（`exists=True` だと英語メッセージになるため自前で検証）
- `-o / --output`: 出力先ファイルパス
- `-m / --model`: モデルパス（デフォルト: `Qwen/Qwen3-ASR-1.7B`）
- `-d / --device`: デバイス指定（デフォルト: 自動検出）
- `-v / --verbose`: 詳細ログ（stderr に出力、stdout はパイプ可能に保つ）

---

### フェーズ 4: テスト

**目的**: CLI とコアロジックの両方についてテストを整備する。

#### 4-1. `tests/conftest.py`

- `Qwen3ASRModel` のモックフィクスチャ（実モデルは約 3.9 GB で CI に不向き）

#### 4-2. `tests/test_cli.py`

| テストケース | 検証内容 |
|---|---|
| `--version` | バージョン文字列が出力される |
| `--help` | ヘルプテキストが出力される |
| 存在しないファイル | エラーメッセージ + 終了コード 1 |
| 非対応拡張子 | 対応フォーマット一覧を含むエラー + 終了コード 1 |
| 正常な `.wav`（モック） | 文字起こし結果が stdout に出力 + 終了コード 0 |
| `--output` 指定 | ファイルが作成され内容が正しい |

#### 4-3. `tests/test_transcriber.py`

| テストケース | 検証内容 |
|---|---|
| `detect_device(None)` CUDA あり | `("cuda:0", torch.bfloat16)` |
| `detect_device(None)` CUDA なし | `("cpu", torch.float32)` |
| `detect_device("cpu")` | `("cpu", torch.float32)` |
| `detect_device("cuda:1")` | `("cuda:1", torch.bfloat16)` |
| `load_model` 失敗時 | `ModelLoadError` が送出される |

---

### フェーズ 5: ドキュメント・仕上げ

- `README.md`: 概要、前提条件、インストール、使い方、開発者向け情報
- `LICENSE`: ライセンス選定（Qwen3-ASR-1.7B のライセンスとの互換性を確認）
- `.gitignore`: Python 標準の除外パターン
- `CLAUDE.md`: ビルド・テストコマンドなどを追記

#### 最終動作確認

```bash
pip install .
qwen3-asr --version
qwen3-asr transcribe sample.wav
qwen3-asr transcribe sample.mp3 -v
qwen3-asr transcribe sample.mp4 -o output.txt
```

---

## 3. 実装順序サマリー

| フェーズ | 作成/変更ファイル | 依存 |
|---|---|---|
| 1: 基盤 | `pyproject.toml`, `__init__.py`, `cli.py`（スケルトン） | なし |
| 2: バリデーション | `exceptions.py`, `cli.py`（バリデーション追加） | フェーズ 1 |
| 3: 推論 | `transcriber.py`, `cli.py`（推論統合） | フェーズ 2 |
| 4: テスト | `conftest.py`, `test_cli.py`, `test_transcriber.py` | フェーズ 3 |
| 5: 仕上げ | `README.md`, `LICENSE`, `.gitignore`, `CLAUDE.md` 更新 | フェーズ 4 |

---

## 4. 技術的な判断事項

### `click.Path(exists=True)` を使わない理由

Click の `exists=True` は英語エラーメッセージを出す。日本語メッセージのために自前で検証し、カスタム例外を使う。

### CPU 実行時の dtype

`torch.bfloat16` は AMD CPU で最大1000倍の性能劣化やハング報告あり。CPU 実行時は必ず `torch.float32` を使用する。`--device cpu` 明示指定時も同様。

### `--device cuda:X` 指定時の dtype

ユーザーが CUDA デバイスを明示指定した場合は公式推奨に従い `torch.bfloat16` を使用する。

### モデルロードのフィードバック

初回は数分かかる可能性がある。非 verbose 時にも `click.echo("モデルを読み込んでいます...", err=True)` を表示する（stderr なのでパイプに影響しない）。

---

## 5. 将来拡張への配慮

- **バッチ処理**: `audio_file` を `nargs=-1` に変更し、ループで処理。
- **タイムスタンプ**: `Qwen3-ForcedAligner-0.6B` 連携、`--timestamps` オプション追加。
- **出力フォーマット**: `--format` オプション追加（JSON / SRT 等）。
