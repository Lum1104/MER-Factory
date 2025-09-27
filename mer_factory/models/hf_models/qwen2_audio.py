import torch
from pathlib import Path
from rich.console import Console
from .base import BaseHFModel
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from typing import List, Dict, Any

console = Console(stderr=True)


class Qwen2AudioModel(BaseHFModel):
    """
    A wrapper for the Qwen2-Audio instruct model from Hugging Face, which is
    specialized for conversational AI tasks involving audio.
    """

    def __init__(self, model_id: str, verbose: bool = True):
        """
        Initializes the Qwen2-Audio model by loading the model and processor.

        Args:
            model_id (str): The ID of the Hugging Face model to load (e.g., 'Qwen/Qwen2-Audio-7B-Instruct').
            verbose (bool): Whether to print verbose logs.
        """
        super().__init__(model_id=model_id, verbose=verbose)
        self.processor = None
        self.model = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads the Hugging Face model and processor for Qwen2-Audio."""
        if self.verbose:
            console.log(f"Initializing Hugging Face pipeline for '{self.model_id}'...")
        try:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            console.log(
                f"Hugging Face model '{self.model_id}' initialized successfully on device: {self.model.device}."
            )
        except Exception as e:
            console.log(
                f"[bold red]ERROR: Could not initialize Hugging Face pipeline: {e}[/bold red]"
            )
            raise

    def _run_generation(self, conversation: List[Dict[str, Any]]) -> str:
        """
        Internal method to run the conversational generation pipeline.
        """
        try:
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele.get("type") == "audio":
                            audios.append(
                                librosa.load(
                                    ele["audio_url"],
                                    sr=self.processor.feature_extractor.sampling_rate,
                                )[0]
                            )

            inputs = self.processor(
                text=text, audios=audios, return_tensors="pt", padding=True
            )

            # Move all tensors to the model's device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Ensure floating point tensors have the correct dtype
            for key, tensor in inputs.items():
                if hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
                    inputs[key] = tensor.to(self.model.dtype)

            with torch.inference_mode():
                generate_ids = self.model.generate(**inputs, max_new_tokens=512)

            # Slice the output to remove the prompt tokens
            input_len = inputs.input_ids.shape[1]
            response_ids = generate_ids[:, input_len:]

            response = self.processor.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response.strip()
        except Exception as e:
            console.log(
                f"[bold red]❌ Error during Qwen2-Audio generation: {e}[/bold red]"
            )
            return f""

    def analyze_audio(self, audio_path: Path, prompt: str) -> dict:
        """
        Analyzes an audio file to produce a transcript using the conversational model.
        """
        if self.verbose:
            console.log(f"Transcribing audio with '{self.model_id}'...")

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        str_response = self._run_generation(conversation)
        return str_response

    def synthesize_summary(self, prompt: str) -> str:
        """Synthesizes a final summary from a text prompt."""
        if self.verbose:
            console.log(f"Synthesizing summary with '{self.model_id}'...")
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._run_generation(conversation)
