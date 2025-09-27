import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from rich.console import Console
from gradio_client import Client
from dotenv import load_dotenv

load_dotenv()
console = Console(stderr=True)


class HFGradioClientModel:
    """
    A client-side proxy that exposes the same interface as local HF models,
    but forwards method calls to a remote Gradio API server that runs the model.

    Expected server contract:
        - POST via Client.predict to api_name "/predict"
        - Payload JSON: { "method": str, "params": dict }
        - Returns JSON string: { "result": Any } or { "error": str }
    """

    def __init__(
        self, model_id: str, verbose: bool = True, base_url: Optional[str] = None
    ):
        self.model_id = model_id
        self.verbose = verbose
        # Prefer explicit arg, then env var, then default localhost
        self.base_url = (
            base_url or os.getenv("HF_API_BASE_URL") or "http://localhost:7860/"
        )

        try:
            self.client = Client(self.base_url)
            if self.verbose:
                console.log(
                    f"Connected HF Gradio client to [cyan]{self.base_url}[/cyan] for model '{self.model_id}'."
                )
        except Exception as e:
            console.log(f"[bold red]Failed to connect to HF API server: {e}[/bold red]")
            raise

    # --- Public API that mirrors local model classes ---
    def analyze_audio(self, audio_path: Path, prompt: str = None) -> Any:
        return self._call_remote(
            "analyze_audio", {"audio_path": str(audio_path), "prompt": prompt}
        )

    def describe_image(self, image_path: Path, prompt: str) -> Any:
        return self._call_remote(
            "describe_image", {"image_path": str(image_path), "prompt": prompt}
        )

    def describe_video(self, video_path: Path, prompt: str = None) -> Any:
        return self._call_remote(
            "describe_video", {"video_path": str(video_path), "prompt": prompt}
        )

    def describe_facial_expression(self, au_text: str) -> Any:
        return self._call_remote("describe_facial_expression", {"au_text": au_text})

    def synthesize_summary(self, prompt: str) -> Any:
        return self._call_remote("synthesize_summary", {"prompt": prompt})

    # --- Internal helpers ---
    def _call_remote(self, method: str, params: Dict[str, Any]) -> Any:
        payload = {"method": method, "params": params}
        payload_str = json.dumps(payload)

        if self.verbose:
            console.log(
                f"Calling remote method [yellow]{method}[/yellow] on {self.base_url}"
            )

        try:
            result_str = self.client.predict(
                payload_str=payload_str, api_name="/predict"
            )
        except Exception as e:
            console.log(
                f"[bold red]Remote call failed for method '{method}': {e}[/bold red]"
            )
            raise

        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            console.log("[bold red]Invalid JSON returned from server.[/bold red]")
            raise

        if "error" in data and data["error"]:
            console.log(
                f"[bold red]Server error for '{method}': {data['error']}[/bold red]"
            )
            raise RuntimeError(data["error"])

        return data.get("result")


if __name__ == "__main__":
    # Lightweight manual test if run directly (uses env HF_API_BASE_URL or localhost)
    model = HFGradioClientModel(
        model_id=os.getenv("HF_MODEL_ID", "openai/whisper-base")
    )
    sample = os.getenv("HF_SAMPLE_AUDIO", str(Path.cwd() / "sample_00000000.wav"))
    try:
        out = model.analyze_audio(Path(sample))
        print(out)
    except Exception as e:
        print(f"Test failed: {e}")
