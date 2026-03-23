from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from qwen3_asr.exceptions import ModelLoadError
from qwen3_asr.transcriber import (
    detect_device,
    format_text,
    load_model,
    split_audio,
    transcribe,
)


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


class TestSplitAudio:
    def test_short_audio_single_chunk(self):
        sr = 16000
        audio = np.zeros(10 * sr)  # 10秒
        chunks = split_audio(audio, sr, chunk_duration=30)
        assert len(chunks) == 1

    def test_long_audio_multiple_chunks(self):
        sr = 16000
        audio = np.zeros(90 * sr)  # 90秒
        chunks = split_audio(audio, sr, chunk_duration=30, overlap=2)
        # 90秒 / (30-2)秒ステップ = ~3.2 → 4チャンク
        assert len(chunks) >= 3

    def test_skip_tiny_tail(self):
        sr = 16000
        # ステップ=28秒。28秒+0.5秒=28.5秒。末尾0.5秒チャンク(< 1秒)はスキップ
        audio = np.zeros(int(28.5 * sr))
        chunks = split_audio(audio, sr, chunk_duration=30, overlap=2)
        assert len(chunks) == 1


class TestFormatText:
    def test_adds_newline_after_period(self):
        assert format_text("あいう。かきく。") == "あいう。\nかきく。"

    def test_no_trailing_newline(self):
        result = format_text("あいう。")
        assert not result.endswith("\n")

    def test_no_period(self):
        assert format_text("あいうかきく") == "あいうかきく"

    def test_empty_string(self):
        assert format_text("") == ""


class TestTranscribe:
    def test_short_audio(self, mocker):
        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.text = "こんにちは"
        mock_model.transcribe.return_value = [fake_result]

        sr = 16000
        short_audio = np.zeros(10 * sr)
        mocker.patch(
            "qwen3_asr.transcriber.load_audio",
            return_value=(short_audio, sr),
        )

        text = transcribe(mock_model, "test.wav")
        assert text == "こんにちは"
        mock_model.transcribe.assert_called_once_with(
            audio="test.wav",
            context="",
            language="Japanese",
        )

    def test_short_audio_with_context(self, mocker):
        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.text = "Qwen3-ASRのテストです"
        mock_model.transcribe.return_value = [fake_result]

        sr = 16000
        short_audio = np.zeros(10 * sr)
        mocker.patch(
            "qwen3_asr.transcriber.load_audio",
            return_value=(short_audio, sr),
        )

        ctx = "用語: Qwen3-ASR, vLLM"
        text = transcribe(mock_model, "test.wav", context=ctx)
        assert text == "Qwen3-ASRのテストです"
        mock_model.transcribe.assert_called_once_with(
            audio="test.wav",
            context=ctx,
            language="Japanese",
        )

    def test_long_audio_chunked(self, mocker):
        mock_model = MagicMock()
        call_count = 0

        def fake_transcribe(audio, context, language):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.text = f"チャンク{call_count}"
            return [result]

        mock_model.transcribe.side_effect = fake_transcribe

        sr = 16000
        long_audio = np.zeros(90 * sr)  # 90秒
        mocker.patch(
            "qwen3_asr.transcriber.load_audio",
            return_value=(long_audio, sr),
        )

        text = transcribe(mock_model, "test.wav")
        assert "チャンク1" in text
        assert call_count >= 3
