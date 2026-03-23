from unittest.mock import MagicMock

import numpy as np
import pytest


class FakeResult:
    def __init__(self, text="テスト音声の文字起こし結果です。"):
        self.text = text
        self.language = "Japanese"


@pytest.fixture
def mock_model(mocker):
    """Qwen3ASRModel のモック + load_audio のモック"""
    mock_cls = mocker.patch("qwen_asr.Qwen3ASRModel")
    mock_instance = MagicMock()
    mock_instance.transcribe.return_value = [FakeResult()]
    mock_cls.from_pretrained.return_value = mock_instance

    # load_audio をモックして短い音声を返す（チャンク分割されないようにする）
    sr = 16000
    short_audio = np.zeros(10 * sr)
    mocker.patch(
        "qwen3_asr.transcriber.load_audio",
        return_value=(short_audio, sr),
    )

    return mock_instance
