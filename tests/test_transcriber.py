from unittest.mock import MagicMock

import pytest
import torch

from qwen3_asr.exceptions import ModelLoadError
from qwen3_asr.transcriber import detect_device, load_model, transcribe


class TestDetectDevice:
    def test_auto_with_cuda(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=True)
        device_map, dtype = detect_device(None)
        assert device_map == "cuda:0"
        assert dtype == torch.bfloat16

    def test_auto_without_cuda(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=False)
        device_map, dtype = detect_device(None)
        assert device_map == "cpu"
        assert dtype == torch.float32

    def test_explicit_cpu(self):
        device_map, dtype = detect_device("cpu")
        assert device_map == "cpu"
        assert dtype == torch.float32

    def test_explicit_cuda(self):
        device_map, dtype = detect_device("cuda:1")
        assert device_map == "cuda:1"
        assert dtype == torch.bfloat16


class TestLoadModel:
    def test_success(self, mock_model, mocker):
        model = load_model("Qwen/Qwen3-ASR-1.7B", "cpu", torch.float32)
        assert model is mock_model

    def test_failure(self, mocker):
        mock_cls = mocker.patch("qwen_asr.Qwen3ASRModel")
        mock_cls.from_pretrained.side_effect = RuntimeError("download failed")

        with pytest.raises(ModelLoadError):
            load_model("Qwen/Qwen3-ASR-1.7B", "cpu", torch.float32)


class TestTranscribe:
    def test_returns_text(self):
        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.text = "こんにちは"
        mock_model.transcribe.return_value = [fake_result]

        text = transcribe(mock_model, "test.wav")
        assert text == "こんにちは"
        mock_model.transcribe.assert_called_once_with(
            audio="test.wav",
            language="Japanese",
        )
