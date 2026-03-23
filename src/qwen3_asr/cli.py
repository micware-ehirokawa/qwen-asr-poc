import click

from qwen3_asr import __version__


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
    click.echo("未実装です", err=True)
