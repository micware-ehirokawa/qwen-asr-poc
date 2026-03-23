# Technical Design Report - Qwen3-ASR におけるコンテキスト注入の技術レポート

## 技術背景

Qwen3-ASR は、従来の音響パターン中心の ASR よりも、LLM の言語知識や world knowledge を活用しやすい Large Audio-Language Model 系の設計で、長文認識、ノイズ耐性、固有表現認識、多言語対応を自然に扱えるようにしています 。技術レポートでは、ASR supervised finetuning の段階で **context biasing data** も利用しつつ、モデルを「prompt 内の自然言語命令には従わない ASR-only モデル」として訓練したと説明しています 。 [modal](https://modal.com/blog/open-source-stt)

その一方で、同じ記述の中で、モデルは **system prompt の context tokens を background knowledge として利用し、customized ASR results を返せる**と明記されています 。つまり Qwen3-ASR のコンテキスト注入は「命令追従」ではなく、「認識候補の事前バイアス付け」に近い仕組みです 。 [modal](https://modal.com/blog/open-source-stt)

## 何を渡すべきか

効果が出やすいのは、会議や収録ごとに出現確率が高い語彙を短く整理したテキストです 。具体的には、人名、会社名、製品名、社内システム名、英字略語、カタカナ語、案件名、議題メモ、関連文書の抜粋が向いています 。 [modal](https://modal.com/blog/open-source-stt)

逆に、「句読点を減らす」「議事録風に直す」「丁寧語に変換する」といった後処理指示は、この仕組みの主目的ではありません 。レポート上も instruction injection や instruction-following failure を避けるために、自然言語命令には従わない設計が選ばれています 。 [modal](https://modal.com/blog/open-source-stt)

## 実装方針

公開 README で確認できる `qwen-asr` の基本例は `Qwen3ASRModel.from_pretrained(...).transcribe(...)` で、例示されている主な入力は `audio`、`language`、`return_time_stamps` などです 。README 上で明示的に system prompt を `transcribe()` に渡す使用例は見当たらない一方、同 README は vLLM ベースの `/v1/chat/completions` と `llm.chat(...)` をサポートしているため、**コンテキスト注入を明示的に扱うなら chat 形式を使うのが最も素直**です 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

README では、Qwen3-ASR 用のローカルサーバーを `qwen-asr-serve` または `vllm serve` で起動し、OpenAI 互換の chat completions API に対して音声を投げる構成が案内されています 。この形なら `system` メッセージに語彙や背景文脈を入れ、`user` メッセージに音声を渡すことで、技術レポートのいう「system prompt の context tokens」を実装上そのまま表現できます 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

## コード例

まずサーバー側は、README にある通り `qwen-asr-serve` で起動できます 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

```bash
pip install -U qwen-asr[vllm]
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.7 --host 0.0.0.0 --port 8000
```

次は OpenAI 互換 Chat Completions を使い、`system` に語彙リストと背景説明を入れる例です 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

context_text = """
これは日本語の社内技術会議です。
出現しやすい語彙:
OpenClaw, LiteLLM, Bedrock, GraphQL, Confluence, Qwen3-ASR,
AmiVoice, Whisper, vLLM, WSL2, RTX A1000, ForcedAligner

案件名:
音声議事録PoC
"""

resp = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    messages=[
        {
            "role": "system",
            "content": context_text
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "file:///data/audio/meeting01.wav"
                    }
                }
            ]
        }
    ],
    temperature=0.01,
)

print(resp.choices[0].message.content)
```

vLLM の Python API を直接使う場合も、README にある `llm.chat(conversation, sampling_params=...)` 形式をそのまま拡張できます 。以下のように `system` を先頭に追加すれば、同じ考え方でコンテキストを渡せます 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-ASR-1.7B")

conversation = [
    {
        "role": "system",
        "content": """
日本語の技術会議。以下を背景知識として使うこと:
人名: 田中, 鈴木
製品名: Qwen3-ASR, Whisper large-v3, AmiVoice
用語: forced alignment, diarization, hallucination, RTF, WER
"""
    },
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": "file:///data/audio/meeting01.wav"}
            }
        ]
    }
]

sampling_params = SamplingParams(
    temperature=0.01,
    max_tokens=512,
)

outputs = llm.chat(conversation, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

## 運用上の注意

コンテキストは「辞書への強制一致」ではなく、あくまで background knowledge として使われるため、入れた語が必ずそのまま出る保証はありません 。そのため、毎回長い文書を丸ごと入れるより、**録音ごとに 10〜50 語程度の高頻度語彙と短い背景説明を与える**ほうが、誤誘導を減らしやすい実装になります 。 [modal](https://modal.com/blog/open-source-stt)

また、Qwen3-ASR は 20 分までの単一音声入力、オフライン/ストリーミング両対応、vLLM ベース推論を想定しているので、実運用では「音声分割 → 文脈付与 → ASR → 必要なら ForcedAligner でタイムスタンプ付与」というパイプラインにすると扱いやすいです 。README ではタイムスタンプ返却には `Qwen3-ForcedAligner-0.6B` を併用する構成が案内されており、これは ASR 本体とは独立した後段処理として組み込めます 。 [zenn](https://zenn.dev/hongbod/articles/def04f586cf168)

