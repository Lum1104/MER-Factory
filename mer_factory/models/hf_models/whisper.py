import torch
from pathlib import Path
from rich.console import Console
from .base import BaseHFModel
from transformers import pipeline
import librosa

console = Console(stderr=True)


class WhisperModel(BaseHFModel):
    """
    A wrapper for the Whisper model from Hugging Face, specialized for
    audio transcription tasks.
    """

    def __init__(self, model_id: str, verbose: bool = True):
        """
        Initializes the Whisper model by loading the ASR pipeline.

        Args:
            model_id (str): The ID of the Hugging Face Whisper model to load (e.g., 'openai/whisper-large-v3').
            verbose (bool): Whether to print verbose logs.
        """
        super().__init__(model_id=model_id, verbose=verbose)
        self.asr_pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads the Hugging Face Whisper ASR pipeline."""
        if self.verbose:
            console.log(
                f"Initializing Hugging Face Whisper ASR pipeline for '{self.model_id}'..."
            )
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                dtype=dtype,
                device=device,
            )

            console.log(
                f"Hugging Face Whisper ASR pipeline '{self.model_id}' initialized successfully on {device}."
            )
        except Exception as e:
            console.log(
                f"[bold red]ERROR: Could not initialize Hugging Face Whisper ASR pipeline: {e}[/bold red]"
            )
            raise

    def _transcribe_audio(self, audio_path: Path) -> str:
        """
        Internal method to transcribe audio using Whisper ASR pipeline.
        """
        try:
            # Load audio file - Whisper pipeline expects 16kHz sampling rate
            audio_array, sampling_rate = librosa.load(str(audio_path), sr=16000)

            # Use the ASR pipeline for transcription
            result = self.asr_pipeline(audio_array)

            # Extract the transcription text
            transcription = result["text"] if isinstance(result, dict) else str(result)

            return transcription.strip()
        except Exception as e:
            console.log(
                f"[bold red]❌ Error during Whisper transcription: {e}[/bold red]"
            )
            return ""

    def analyze_audio(self, audio_path: Path, prompt: str = None) -> str:
        """
        Transcribes an audio file using Whisper.

        Args:
            audio_path (Path): Path to the audio file (WAV format).
            prompt (str, optional): Not used for Whisper transcription, included for interface compatibility.

        Returns:
            str: The transcribed text from the audio.
        """
        if self.verbose:
            console.log(f"Transcribing audio with Whisper model '{self.model_id}'...")

        return self._transcribe_audio(audio_path)
