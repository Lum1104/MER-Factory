from pathlib import Path
from rich.console import Console


console = Console(stderr=True)


class BaseHFModel:
    """
    Base class for HF models with default no-op implementations.
    Subclasses should override supported capabilities.
    """

    def __init__(self, model_id: str, verbose: bool = True):
        self.model_id = model_id
        self.verbose = verbose

    def analyze_audio(self, audio_path: Path, prompt: str = None):
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support analyze_audio.[/yellow]"
            )
        return ""

    def describe_image(self, image_path: Path, prompt: str = None):
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support describe_image.[/yellow]"
            )
        return ""

    def describe_video(self, video_path: Path, prompt: str = None):
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support describe_video.[/yellow]"
            )
        return ""

    def describe_facial_expression(self, au_text: str):
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support describe_facial_expression.[/yellow]"
            )
        return ""

    def synthesize_summary(self, prompt: str):
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support synthesize_summary.[/yellow]"
            )
        return ""
