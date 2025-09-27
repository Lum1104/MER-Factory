import torch
from pathlib import Path
from rich.console import Console
from .base import BaseHFModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from typing import List, Dict, Any

try:
    from qwen_omni_utils import process_mm_info
except ImportError as e:
    raise ImportError(
        "Failed to import qwen_omni_utils. "
        "Please ensure you have the 'qwen-omni-utils' package installed. "
        "You can install it using: pip install qwen-omni-utils"
    ) from e

console = Console(stderr=True)


class Qwen2_5OmniModel(BaseHFModel):
    """
    A wrapper for the Qwen/Qwen2.5-Omni-7B model, which is a multimodal model
    capable of processing text, audio, images, and video.
    """

    def __init__(self, model_id: str, verbose: bool = True):
        """
        Initializes the Qwen2.5-Omni model.

        Args:
            model_id (str): The ID of the Hugging Face model.
            verbose (bool): Whether to print verbose logs.
        """
        super().__init__(model_id=model_id, verbose=verbose)
        self.processor = None
        self.model = None
        self.system_prompt = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        }

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads the Hugging Face model and processor for Qwen2.5-Omni."""
        if self.verbose:
            console.log(f"Initializing Hugging Face pipeline for '{self.model_id}'...")
        try:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
            console.log(
                f"Hugging Face model '{self.model_id}' initialized successfully on device: {self.model.device}."
            )
        except Exception as e:
            console.log(
                f"[bold red]ERROR: Could not initialize Hugging Face pipeline: {e}[/bold red]"
            )
            raise

    def _run_generation(
        self, conversation: List[Dict[str, Any]], use_audio_in_video: bool = True
    ):
        """
        Internal method to run the generation pipeline.
        It generates both text and potentially audio.
        """
        try:
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=use_audio_in_video
            )

            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio_in_video,
            )

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            for key, tensor in inputs.items():
                if hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
                    inputs[key] = tensor.to(self.model.dtype)

            with torch.inference_mode():
                # The model can generate both text and audio.
                text_ids, audio_output = self.model.generate(
                    **inputs, use_audio_in_video=use_audio_in_video, max_new_tokens=512
                )
            input_len = inputs.input_ids.shape[1]
            response_ids = text_ids[:, input_len:]

            generated_text = self.processor.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return generated_text.strip()

        except Exception as e:
            console.log(
                f"[bold red]❌ Error during Qwen2.5-Omni generation: {e}[/bold red]"
            )
            return f""

    def describe_facial_expression(self, prompt: str) -> str:
        """Generates a description from AU text."""

        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._run_generation(conversation)

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """Generates a description for an image file."""

        conversation = [
            self.system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self._run_generation(conversation)

    def analyze_audio(self, audio_path: Path, prompt: str) -> dict:
        """Analyzes an audio file and returns a structured dictionary."""
        conversation = [
            self.system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        str_response = self._run_generation(conversation)
        return str_response

    def describe_video(self, video_path: Path, prompt: str) -> str:
        """Generates a description for a video."""
        conversation = [
            self.system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self._run_generation(conversation, use_audio_in_video=True)

    def synthesize_summary(self, prompt: str) -> str:
        """Synthesizes a final summary from a text prompt."""
        conversation = [
            self.system_prompt,
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._run_generation(conversation)
