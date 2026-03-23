# Technical Design Report - Qwen3-ASR-1.7B を Python で使う方法

Qwen3-ASR-1.7B は、まず Python 環境を作って `qwen-asr` を入れ、次に `Qwen3ASRModel.from_pretrained()` でモデルを読み込んで `transcribe()` を呼ぶ、という流れです 。日本語は対応言語に含まれており、ローカルパス・URL・base64・`(np.ndarray, sr)` 形式の音声入力をそのまま渡せます 。 [huggingface](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

## 最短手順

公式の最小構成は、**Python 3.12 の新しい環境**を作成して `pip install -U qwen-asr` を実行するやり方です 。より高速な推論やストリーミングを使いたい場合は `pip install -U qwen-asr[vllm]`、長い音声や大きめのバッチで GPU メモリ効率を上げたい場合は `flash-attn` の追加も推奨されています 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U qwen-asr

# 高速化したい場合
# pip install -U qwen-asr[vllm]
# pip install -U flash-attn --no-build-isolation
```

## Python 実行例

transformers バックエンドの基本形は、`dtype=torch.bfloat16` と `device_map="cuda:0"` を指定してモデルを読み込み、`transcribe()` に音声ファイルを渡すだけです 。言語は `language=None` で自動判定できますが、日本語固定にしたいなら `"Japanese"` を指定するのが分かりやすいです 。 [huggingface](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=8,
    max_new_tokens=512,
)

results = model.transcribe(
    audio="sample_ja.wav",     # ローカル音声ファイル
    language="Japanese",       # 自動判定なら None
)

print(results[0].language)
print(results[0].text)
```

## タイムスタンプ付き

単語や文字単位のタイムスタンプが必要なら、`Qwen3-ForcedAligner-0.6B` を一緒に読み込んで `return_time_stamps=True` を付けます 。Forced Aligner は日本語を含む 11 言語に対応すると案内されています 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=8,
    max_new_tokens=512,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs={
        "dtype": torch.bfloat16,
        "device_map": "cuda:0",
    },
)

results = model.transcribe(
    audio="sample_ja.wav",
    language="Japanese",
    return_time_stamps=True,
)

print(results[0].text)
print(results[0].time_stamps[:3])
```

## サーバー運用

最速寄りで使うなら、公式は vLLM バックエンドを推奨しており、`Qwen3ASRModel.LLM(...)` か `qwen-asr-serve` / `vllm serve` でサーバー化できます 。ストリーミング推論は現状 vLLM バックエンドのみ対応で、タイムスタンプ返却とは同時利用できないと案内されています 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

```bash
pip install -U qwen-asr[vllm]
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

## 実運用の注意

モデル読み込み時には、モデル名指定で重みを自動ダウンロードできますが、実行環境で外部ダウンロードできない場合は `huggingface-cli download` や `modelscope download` で事前取得できます 。公式の評価例では BF16 を使っており、FlashAttention 2 も `torch.float16` または `torch.bfloat16` での利用が前提です 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

