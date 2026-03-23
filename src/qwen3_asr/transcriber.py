import torch

from qwen3_asr.exceptions import ModelLoadError


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


def load_model(model_path: str, device_map: str, dtype: torch.dtype):
    """Qwen3-ASR モデルをロードする"""
    try:
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
        )
        return model
    except Exception as e:
        raise ModelLoadError(model_path, e) from e


def transcribe(model, audio_path: str) -> str:
    """音声ファイルを文字起こしする"""
    results = model.transcribe(
        audio=audio_path,
        language="Japanese",
    )
    return results[0].text
