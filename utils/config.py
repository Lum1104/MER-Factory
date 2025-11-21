import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, FilePath, DirectoryPath, field_validator

# Load environment variables from .env file
load_dotenv()

# Define constants for file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


class TaskType(str, Enum):
    """Enum for the different types of analysis tasks that can be performed."""

    EMOTION_RECOGNITION = "MERR"
    SENTIMENT_ANALYSIS = "Sentiment Analysis"


class ProcessingType(str, Enum):
    """Enum for the different types of processing that can be performed."""

    AU = "AU"
    AUDIO = "audio"
    VIDEO = "video"
    MER = "MER"
    IMAGE = "image"


# Maps processing types to the suffix of their final output file.
FINAL_OUTPUT_FILENAMES = {
    ProcessingType.MER: "_merr_data.json",
    ProcessingType.AU: "_au_analysis.json",
    ProcessingType.AUDIO: "_audio_analysis.json",
    ProcessingType.VIDEO: "_video_analysis.json",
    ProcessingType.IMAGE: "_image_analysis.json",
}


class AppConfig(BaseModel):
    """
    A Pydantic model to hold and validate all application settings.
    """

    input_path: Path
    output_dir: DirectoryPath
    processing_type: ProcessingType
    task: TaskType = Field(
        TaskType.EMOTION_RECOGNITION,
        description="The analysis task to perform.",
    )
    prompts_file: Path = Field(
        Path("utils/prompts/prompts.json"), description="Path to the prompts JSON file."
    )
    error_logs_dir: Path
    label_file: Optional[FilePath] = None
    threshold: float = Field(0.8, ge=0.0, le=5.0)
    peak_distance_frames: int = Field(15, ge=8)
    silent: bool = False
    cache: bool = False
    concurrency: int = Field(4, ge=1)
    use_gate_agent: bool = False
    ollama_vision_model: Optional[str] = None
    ollama_text_model: Optional[str] = None
    chatgpt_model: Optional[str] = None
    huggingface_model_id: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)

    # Private attributes for environment variables
    openface_executable: Optional[str] = Field(
        default=os.getenv("OPENFACE_EXECUTABLE"), repr=False
    )
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"), repr=False
    )
    google_api_key: Optional[str] = Field(
        default=os.getenv("GOOGLE_API_KEY"), repr=False
    )

    def __init__(self, **data):
        """
        Override init to create directories before validation and to
        dynamically create the error_logs_dir path.
        """
        # Manually handle directory creation before Pydantic validation.
        if "output_dir" in data:
            output_dir_path = Path(data["output_dir"])
            error_logs_path = output_dir_path / "error_logs"

            # Create directories *before* validation.
            output_dir_path.mkdir(parents=True, exist_ok=True)
            error_logs_path.mkdir(exist_ok=True)

            # Add the calculated error_logs_dir path to the data.
            data["error_logs_dir"] = error_logs_path

        super().__init__(**data)

    @field_validator("input_path")
    def validate_input_path(cls, v: Path) -> Path:
        """Validate that the input path exists."""
        if not v.exists():
            raise ValueError(f"Input path does not exist: {v}")
        return v

    @property
    def verbose(self) -> bool:
        """Derived property for verbosity."""
        return not self.silent

    @property
    def api_key(self) -> Optional[str]:
        """Returns the appropriate API key based on the selected model."""
        if self.chatgpt_model:
            return self.openai_api_key
        return self.google_api_key

    def get_model_choice_error(self) -> Optional[str]:
        """Check if at least one model is configured."""
        if not any(
            [
                self.huggingface_model_id,
                self.ollama_text_model,
                self.ollama_vision_model,
                self.chatgpt_model,
                self.api_key and not self.chatgpt_model,  # Gemini
            ]
        ):
            return "A model must be provided via --huggingface-model, --ollama-..., --chatgpt-model, or a GOOGLE_API_KEY/OPENAI_API_KEY in the .env file."
        return None

    def get_openface_path_error(self) -> Optional[str]:
        """Check if OpenFace is needed and if the path is valid."""
        if self.processing_type in [ProcessingType.MER, ProcessingType.AU]:
            if (
                not self.openface_executable
                or not Path(self.openface_executable).exists()
            ):
                return f"Error: OpenFace executable not found at '{self.openface_executable}'. Please check your .env file."
        return None
