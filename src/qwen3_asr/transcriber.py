import numpy as np
import torch

from qwen3_asr.exceptions import ModelLoadError

# 音声チャンクの長さ（秒）
CHUNK_DURATION = 30
# チャンク間のオーバーラップ（秒）- 文境界の切断を軽減
CHUNK_OVERLAP = 2


def detect_device(device: str | None) -> tuple[str, torch.dtype]:
    """デバイスと dtype を決定する

    Args:
        device: ユーザー指定のデバイス文字列。None の場合は自動検出。

    Returns:
        (device_map, dtype) のタプル
    """
    if device is not None:
        if device.startswith("cuda"):
            return device, torch.bfloat16
        return device, torch.float32

    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16

    return "cpu", torch.float32


def load_model(
    model_path: str,
    device_map: str,
    dtype: torch.dtype,
    max_new_tokens: int = 2048,
    max_inference_batch_size: int = 8,
):
    """Qwen3-ASR モデルをロードする"""
    try:
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=max_inference_batch_size,
        )
        return model
    except Exception as e:
        raise ModelLoadError(model_path, e) from e


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """音声ファイルを読み込む"""
    import librosa

    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr


def split_audio(
    audio: np.ndarray,
    sr: int,
    chunk_duration: int = CHUNK_DURATION,
    overlap: int = CHUNK_OVERLAP,
) -> list[np.ndarray]:
    """音声を指定秒数のチャンクに分割する"""
    chunk_samples = chunk_duration * sr
    step_samples = (chunk_duration - overlap) * sr
    chunks = []

    for start in range(0, len(audio), step_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        # 極端に短いチャンクはスキップ
        if len(chunk) < sr:
            break
        chunks.append(chunk)

    return chunks


def format_text(text: str) -> str:
    """句点で改行を挿入する"""
    return text.replace("。", "。\n").rstrip("\n")


def transcribe(
    model, audio_path: str, context: str = "", verbose_callback=None
) -> str:
    """音声ファイルを文字起こしする

    長時間音声は自動的にチャンク分割して処理する。
    context を指定すると、バックグラウンド知識として認識精度を向上させる。
    """
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr

    # 短い音声はそのまま処理
    if duration <= CHUNK_DURATION:
        results = model.transcribe(
            audio=audio_path, context=context, language="Japanese"
        )
        return format_text(results[0].text)

    # 長い音声はチャンク分割
    chunks = split_audio(audio, sr)
    if verbose_callback:
        verbose_callback(f"音声を {len(chunks)} チャンクに分割しました（{duration:.0f}秒）")

    texts = []
    for i, chunk in enumerate(chunks):
        if verbose_callback:
            verbose_callback(f"チャンク {i + 1}/{len(chunks)} を処理中...")
        results = model.transcribe(
            audio=(chunk, sr), context=context, language="Japanese"
        )
        texts.append(results[0].text)

    return format_text("".join(texts))
