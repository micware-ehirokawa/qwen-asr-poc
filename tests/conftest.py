from unittest.mock import MagicMock

import pytest


class FakeResult:
    def __init__(self, text="テスト音声の文字起こし結果です。"):
        self.text = text
        self.language = "Japanese"


@pytest.fixture
def mock_model(mocker):
    """Qwen3ASRModel のモック"""
    mock_cls = mocker.patch("qwen_asr.Qwen3ASRModel")
    mock_instance = MagicMock()
    mock_instance.transcribe.return_value = [FakeResult()]
    mock_cls.from_pretrained.return_value = mock_instance
    return mock_instance
