import shutil
import sys
from pathlib import Path

import click

from qwen3_asr import __version__
from qwen3_asr.exceptions import (
    AudioFileNotFoundError,
    FfmpegNotFoundError,
    Qwen3AsrError,
    UnsupportedFormatError,
)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".mp4"}


def validate_audio_file(audio_file: str) -> Path:
    """音声ファイルのバリデーションを行う"""
    path = Path(audio_file)

    if not path.exists():
        raise AudioFileNotFoundError(audio_file)

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(audio_file, suffix)

    if suffix in {".mp3", ".mp4"} and shutil.which("ffmpeg") is None:
        raise FfmpegNotFoundError()

    return path


@click.group()
@click.version_option(version=__version__, prog_name="qwen3-asr")
def main():
    """Qwen3-ASR-1.7B を使った日本語音声文字起こしツール"""
    pass


@main.command()
@click.argument("audio_file")
@click.option("-o", "--output", default=None, type=click.Path(), help="結果をファイルに出力")
@click.option("-m", "--model", default="Qwen/Qwen3-ASR-1.7B", help="モデルディレクトリの指定")
@click.option("-d", "--device", default=None, help="デバイス指定 (cpu / cuda:0 など)")
@click.option("-v", "--verbose", is_flag=True, help="詳細ログを表示")
def transcribe(audio_file, output, model, device, verbose):
    """音声ファイルを日本語テキストに文字起こしする"""
    try:
        path = validate_audio_file(audio_file)

        # TODO: フェーズ 3 でモデル推論を実装
        click.echo("未実装です", err=True)

    except Qwen3AsrError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
