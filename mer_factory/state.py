from pathlib import Path
from typing import TypedDict, Dict, Any, List

from .models import LLMModels
from .prompts import PromptTemplates


class MERRState(TypedDict, total=False):
    """
    Represents the incrementally built state of the MERR pipeline.
    `total=False` allows the state to be valid even when only partially populated
    as it flows through the graph.
    """

    # === CORE CONFIGURATION & SETUP ===
    processing_type: str  # The type of pipeline to run (e.g., 'MER', 'audio').
    task: str  # The analysis task to perform (e.g., 'MERR').
    models: LLMModels  # An instance of the LLMModels class for API calls.
    prompts: PromptTemplates  # An instance of the PromptTemplates class.
    verbose: bool  # Flag for detailed logging output.
    cache: bool  # Flag to reuse existing analysis results from previous runs.
    ground_truth_label: str  # Optional ground truth label for the media file.

    # === FILE & PATH MANAGEMENT ===
    video_path: Path  # Path to the source video or image file.
    video_id: str  # The unique identifier (stem) of the source file.
    output_dir: Path  # The main output directory for all processing.
    video_output_dir: Path  # The specific output subdirectory for the current file.
    error_logs_dir: Path  # Directory to store error logs.
    audio_path: Path  # Path to the extracted audio file (.wav).
    au_data_path: Path  # Path to the OpenFace Action Unit data (.csv).
    peak_frame_path: Path  # Path to the saved peak emotional frame image.

    # === FACIAL & EMOTION ANALYSIS RESULTS ===
    threshold: float  # Confidence threshold for emotion detection.
    peak_distance_frames: int  # Minimum distance between emotional peaks.
    detected_emotions: List  # A chronological list of detected emotion summaries.
    peak_frame_info: Dict[
        str, Any
    ]  # Detailed information about the overall peak frame.
    peak_frame_au_description: str  # Text description of AUs at the peak frame.

    # === MULTIMODAL DESCRIPTION RESULTS ===
    audio_analysis_results: str  # LLM-generated summary of the audio content.
    video_description: str  # LLM-generated summary of the overall video content.
    image_visual_description: (
        str  # LLM-generated description of the peak frame's visual content.
    )
    descriptions: Dict[
        str, str
    ]  # A dictionary compiling all coarse descriptions before final synthesis.

    # === IMAGE-ONLY PIPELINE RESULTS ===
    au_text_description: str  # Raw text summary of AUs detected in the image.
    llm_au_description: (
        str  # LLM-generated interpretation of the facial expression from AUs.
    )

    # === FINAL SYNTHESIS & ERROR HANDLING ===
    final_summary: str  # The final, synthesized multimodal summary from the LLM.
    error: str  # An error message, if any node in the graph fails.
