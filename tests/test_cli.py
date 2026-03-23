import pytest
from click.testing import CliRunner

from qwen3_asr.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestVersionAndHelp:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output

    def test_transcribe_help(self, runner):
        result = runner.invoke(main, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output


class TestValidation:
    def test_nonexistent_file(self, runner):
        result = runner.invoke(main, ["transcribe", "nonexistent.wav"])
        assert result.exit_code == 1
        assert "音声ファイルが見つかりません" in result.output

    def test_unsupported_format(self, runner, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("dummy")
        result = runner.invoke(main, ["transcribe", str(txt_file)])
        assert result.exit_code == 1
        assert "非対応のファイル形式です" in result.output
        assert ".wav" in result.output


class TestTranscribe:
    def test_stdout_output(self, runner, tmp_path, mock_model):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        result = runner.invoke(main, ["transcribe", str(wav_file)])
        assert result.exit_code == 0
        assert "テスト音声の文字起こし結果です。" in result.output

    def test_file_output(self, runner, tmp_path, mock_model):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)
        out_file = tmp_path / "result.txt"

        result = runner.invoke(
            main, ["transcribe", str(wav_file), "-o", str(out_file)]
        )
        assert result.exit_code == 0
        assert out_file.read_text() == "テスト音声の文字起こし結果です。"

    def test_verbose(self, runner, tmp_path, mock_model):
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        result = runner.invoke(main, ["transcribe", str(wav_file), "-v"])
        assert result.exit_code == 0
        assert "デバイス:" in result.output
        assert "モデルの読み込みが完了しました" in result.output
