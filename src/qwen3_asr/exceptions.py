class Qwen3AsrError(Exception):
    """qwen3-asr の基底例外クラス"""
    pass


class AudioFileNotFoundError(Qwen3AsrError):
    def __init__(self, path: str):
        super().__init__(f"音声ファイルが見つかりません: {path}")


class UnsupportedFormatError(Qwen3AsrError):
    SUPPORTED = [".wav", ".mp3", ".mp4"]

    def __init__(self, path: str, suffix: str):
        supported = ", ".join(self.SUPPORTED)
        super().__init__(
            f"非対応のファイル形式です: {suffix}\n"
            f"対応フォーマット: {supported}\n"
            f"ファイル: {path}"
        )


class ModelLoadError(Qwen3AsrError):
    def __init__(self, model_path: str, cause: Exception):
        super().__init__(
            f"モデルの読み込みに失敗しました: {model_path}\n"
            f"原因: {cause}\n"
            f"ネットワーク接続を確認するか、--model でローカルパスを指定してください。"
        )


class FfmpegNotFoundError(Qwen3AsrError):
    def __init__(self):
        super().__init__(
            "ffmpeg が見つかりません。mp3/mp4 の処理には ffmpeg が必要です。\n"
            "インストール方法:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
