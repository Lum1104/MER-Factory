---
layout: default
title: API Reference
description: Complete API reference for MER-Factory classes and functions
lang: en
---

# API Reference

Complete reference documentation for MER-Factory's core classes, functions, and modules.

## Core Modules

### `main.py` - CLI Entry Point

The main entry point for the MER-Factory command-line interface, built with Typer.

#### Functions

##### `main_orchestrator(config: AppConfig) -> None`

The main orchestration function that coordinates the entire processing pipeline. It is called by the `process` command.

**Parameters:**

* `config` (`AppConfig`): An application configuration object that contains all processing parameters derived from the CLI arguments.

**Process:**

1. Validates model choices and required paths.

2. Loads ground truth labels if a label file is provided.

3. Initializes the required LLM models.

4. Discovers all media files to be processed in the input path.

5. **Phase 1:** Runs the feature extraction phase (`FFmpeg` for audio, `OpenFace` for facial Action Units) concurrently.

6. **Phase 2:** Executes the main processing pipeline using a `LangGraph` graph.

7. Handles concurrent processing for multiple files, with support for both synchronous (Hugging Face) and asynchronous (OpenAI, Ollama) models.

8. Prints a final summary of successful, skipped, and failed files.

##### `process(...)`

The Typer command that serves as the user-facing entry point for the CLI. It gathers all user inputs and passes them to the `main_orchestrator`.

**Arguments & Options:**

* `input_path` (Path): **\[Required\]** Path to a single media file or a directory containing media files.

* `output_dir` (Path): **\[Required\]** Directory where all outputs (CSVs, logs, etc.) will be saved.

* `--type` / `-t` (ProcessingType): The type of processing to perform. Defaults to `MER` (Multimodal Emotion Recognition).

* `--label-file` / `-l` (Path): Optional path to a CSV file containing `name` and `label` columns for ground truth data.

* `--threshold` / `-th` (float): The intensity threshold for detecting an emotion from Action Units. Defaults to `0.8`.

* `--peak_dis` / `-pd` (int): The minimum distance (in frames) between two detected emotional peaks. Defaults to `15`.

* `--concurrency` / `-c` (int): The number of files to process concurrently. Defaults to `4`.

* `--ollama-vision-model` / `-ovm` (str): The name of the Ollama vision model to use (e.g., `llava`).

* `--ollama-text-model` / `-otm` (str): The name of the Ollama text model to use (e.g., `llama3`).

* `--chatgpt-model` / `-cgm` (str): The name of the OpenAI model to use (e.g., `gpt-4o`).

* `--huggingface-model` / `-hfm` (str): The model ID of the Hugging Face model to use.

* `--silent` / `-s` (bool): If set, runs the CLI with minimal console output. Defaults to `False`.

* `--cache` / `-ca` (bool): If set, reuses existing feature extraction results from previous runs. Defaults to `False`.

### `facial_analyzer.py` - Facial Analysis

This module contains the `FacialAnalyzer` class, which is responsible for processing the output from OpenFace.

#### Class: `FacialAnalyzer`

Analyzes a full OpenFace Action Unit (AU) data file to find emotional peaks, identify the most intense overall frame, and generate summaries.

##### `__init__(self, au_data_path: Path)`

Initializes the analyzer by loading and pre-processing the OpenFace CSV file.

* **Parameters:**

  * `au_data_path` (Path): The path to the OpenFace output CSV file.

* **Raises:**

  * `FileNotFoundError`: If the CSV file does not exist.

  * `ValueError`: If the CSV is empty or contains no AU intensity columns (`_r` suffix).

##### `get_chronological_emotion_summary(self, peak_height=0.8, peak_distance=20, emotion_threshold=0.8)`

Finds all significant emotional peaks in the video's timeline and generates a human-readable summary for each.

* **Parameters:**

  * `peak_height` (float): The minimum "overall intensity" required to register as a peak. Defaults to `0.8`.

  * `peak_distance` (int): The minimum number of frames between peaks. Defaults to `20`.

  * `emotion_threshold` (float): The minimum AU intensity score to consider an emotion present at a peak. Defaults to `0.8`.

* **Returns:**

  * `tuple`: A tuple containing:

    * `list`: A list of strings, where each string describes an emotional peak (e.g., `"Peak at 3.45s: happy (moderate), surprise (slight)"`). Returns `["neutral"]` if no peaks are found.

    * `bool`: A flag indicating if the video was considered expressive (i.e., if any valid emotional peaks were detected).

##### `get_overall_peak_frame_info(self)`

Finds the single most emotionally intense frame in the entire video based on the sum of AU intensities.

* **Returns:**

  * `dict`: A dictionary containing information about the peak frame, including `frame_number`, `timestamp`, and a dictionary of the most active AUs and their intensities (`top_aus_intensities`).

##### `get_frame_au_summary(self, frame_index=0, threshold=0.8)`

Gets the Action Unit summary for a specific frame. This is primarily used for single-image analysis.

* **Parameters:**

  * `frame_index` (int): The index of the frame to analyze. Defaults to `0`.

  * `threshold` (float): The minimum intensity for an AU to be considered active. Defaults to `0.8`.

* **Returns:**

  * `str`: A human-readable description of the active AUs in the specified frame.

### `emotion_analyzer.py` - Emotion and AU Logic

This module provides the core logic for interpreting Action Units (AUs) as emotions, based on established mappings.

#### Class: `EmotionAnalyzer`

A utility class that provides static methods to centralize the logic for analyzing emotions and Action Units from OpenFace data.

##### `analyze_emotions_at_peak(peak_frame, emotion_threshold=0.8)`

Analyzes the emotions present in a single data frame (typically an emotional peak).

* **Parameters:**

  * `peak_frame` (pd.Series): A row from a Pandas DataFrame corresponding to a peak frame.

  * `emotion_threshold` (float): The minimum average intensity score for an emotion's constituent AUs to be considered present. Defaults to `0.8`.

* **Returns:**

  * `list`: A list of dictionaries. Each dictionary contains details of a detected emotion, including `emotion`, `score`, and `strength` (slight, moderate, or strong).

##### `get_active_aus(frame_data, threshold=0.8)`

Extracts a dictionary of active Action Units and their intensities from a single frame's data.

* **Parameters:**

  * `frame_data` (pd.Series): A row from a DataFrame.

  * `threshold` (float): The minimum intensity for an AU to be considered active. Defaults to `0.8`.

* **Returns:**

  * `dict`: A dictionary mapping active AU codes (e.g., `AU06_r`) to their intensity values.

##### `extract_au_description(active_aus)`

Generates a human-readable string from a dictionary of active AUs.

* **Parameters:**

  * `active_aus` (dict): A dictionary of active AU codes and their intensities, as returned by `get_active_aus`.

* **Returns:**

  * `str`: A formatted string describing the active AUs and their intensities (e.g., "Cheek raiser (intensity: 2.50), Lip corner puller (smile) (intensity: 1.80)"). Returns "Neutral expression." if the input dictionary is empty.

#### Constants

* `AU_TO_TEXT_MAP`: A dictionary that maps OpenFace Action Unit codes (e.g., `"AU01_r"`) to human-readable descriptions (e.g., `"Inner brow raiser"`).

* `EMOTION_TO_AU_MAP`: A dictionary that maps basic emotions (e.g., `"happy"`) to a list of the primary Action Units (presence, `_c`) that constitute that emotion. This is the core mapping used for emotion detection.

---

## Utility Modules

### `utils/config.py` - Application Configuration

This module defines the main configuration structure for the application using Pydantic, ensuring that all settings are validated and correctly typed.

#### Class: `AppConfig`

A Pydantic `BaseModel` that holds and validates all application settings passed from the CLI.

**Attributes:**

* `input_path` (Path): Path to the input file or directory.

* `output_dir` (DirectoryPath): Path to the directory where results will be saved.

* `processing_type` (ProcessingType): An enum indicating the type of analysis to perform (`MER`, `AU`, `audio`, etc.).

* `error_logs_dir` (Path): Path to the directory for storing error logs, created automatically within the `output_dir`.

* `label_file` (Optional\[FilePath]): Optional path to a CSV file with ground truth labels.

* `threshold` (float): Intensity threshold for emotion detection.

* `peak_distance_frames` (int): Minimum distance in frames between detected emotional peaks.

* `silent` (bool): Flag to suppress detailed console output.

* `cache` (bool): Flag to enable caching of intermediate results.

* `concurrency` (int): Number of files to process in parallel.

* `ollama_vision_model` (Optional\[str]): Name of the Ollama vision model to use.

* `ollama_text_model` (Optional\[str]): Name of the Ollama text model to use.

* `chatgpt_model` (Optional\[str]): Name of the OpenAI model to use.

* `huggingface_model_id` (Optional\[str]): ID of the Hugging Face model to use.

* `labels` (Dict\[str, str]): A dictionary to store loaded ground truth labels.

* `openface_executable` (Optional\[str]): Path to the OpenFace executable, loaded from the `.env` file.

* `openai_api_key` (Optional\[str]): OpenAI API key, loaded from the `.env` file.

* `google_api_key` (Optional\[str]): Google API key, loaded from the `.env` file.

**Methods:**

* `get_model_choice_error() -> Optional[str]`: Validates that at least one LLM has been configured.

* `get_openface_path_error() -> Optional[str]`: Checks if the OpenFace executable path is valid when required.

#### Enum: `ProcessingType`

A string `Enum` that defines the valid types of processing that can be performed.

**Members:**

* `AU`: Action Unit analysis only.

* `AUDIO`: Audio analysis only.

* `VIDEO`: Video description only.

* `MER`: Full Multimodal Emotion Recognition.

* `IMAGE`: Image analysis only.

### `utils/file_handler.py` - File System Operations

This module contains helper functions for discovering files and loading data from the file system.

#### Functions

##### `find_files_to_process(input_path: Path, verbose: bool = True) -> List[Path]`

Recursively finds all valid media files (video, image, audio) from a given input path.

* **Parameters:**

  * `input_path` (Path): The path to a single file or a directory.

  * `verbose` (bool): If `True`, prints detailed output.

* **Returns:**

  * `List[Path]`: A list of `Path` objects for all files to be processed.

* **Raises:**

  * `SystemExit`: If the input path is invalid or no processable files are found.

##### `load_labels_from_file(label_file: Path, verbose: bool = True) -> Dict[str, str]`

Loads ground truth labels from a specified CSV file. The CSV must contain `name` and `label` columns.

* **Parameters:**

  * `label_file` (Path): Path to the CSV file.

  * `verbose` (bool): If `True`, prints detailed output.

* **Returns:**

  * `Dict[str, str]`: A dictionary mapping a file's stem (name without extension) to its ground truth label.

* **Raises:**

  * `SystemExit`: If the file is not found or cannot be parsed correctly.

### `utils/processing_manager.py` - Processing and Feature Extraction

This module manages the execution flow of the main processing tasks, including feature extraction and handling concurrency.

#### Functions

##### `build_initial_state(file_path: Path, config: AppConfig, models: LLMModels) -> MERRState`

Constructs the initial state dictionary for a single file that will be passed to the processing graph.

* **Parameters:**

  * `file_path` (Path): The path to the media file being processed.

  * `config` (AppConfig): The application configuration object.

  * `models` (LLMModels): The initialized LLM models object.

* **Returns:**

  * `MERRState`: A TypedDict containing the initial state for the graph.

##### `run_main_processing(...) -> Dict[str, int]`

Manages the main processing loop. It checks for cached final outputs and dispatches jobs to be run either synchronously or asynchronously based on the model's capabilities.

* **Parameters:**

  * `files_to_process` (List\[Path]): A list of files to process.

  * `graph_app` (Any): The compiled `langgraph` application.

  * `initial_state_builder` (functools.partial): A partial function for building the initial state.

  * `config` (AppConfig): The application configuration object.

  * `is_sync` (bool): A flag indicating whether to run in synchronous mode.

* **Returns:**

  * `Dict[str, int]`: A dictionary summarizing the results (`success`, `failure`, `skipped`).

##### `run_feature_extraction(files_to_process: List[Path], config: AppConfig)`

Asynchronously runs the necessary feature extraction tools (`FFmpeg` for audio, `OpenFace` for facial AUs) on the files before the main processing begins.

* **Parameters:**

  * `files_to_process` (List\[Path]): A list of files needing feature extraction.

  * `config` (AppConfig): The application configuration object.

---

## Agent & Graph Modules

### `mer_factory/state.py` - Pipeline State

This module defines the shared state object that is passed between all nodes in the processing graph.

#### Class: `MERRState`

A `TypedDict` that represents the incrementally built state of the MERR pipeline. Using `total=False` allows the state to be valid even when only partially populated as it flows through the graph. The state is organized into the following sections:

* **Core Configuration & Setup:** `processing_type`, `models`, `verbose`, `cache`, `ground_truth_label`.
* **File & Path Management:** `video_path`, `video_id`, `output_dir`, `audio_path`, `au_data_path`, `peak_frame_path`, etc.
* **Facial & Emotion Analysis Results:** `threshold`, `peak_distance_frames`, `detected_emotions`, `peak_frame_info`, `peak_frame_au_description`.
* **Multimodal Description Results:** `audio_analysis_results`, `video_description`, `image_visual_description`.
* **Image-Only Pipeline Results:** `au_text_description`, `llm_au_description`.
* **Final Synthesis & Error Handling:** `final_summary`, `error`.

### `mer_factory/graph.py` - Processing Graph

This module constructs the processing pipeline using `langgraph.StateGraph`. It defines all the nodes, edges, and conditional routing logic.

#### Functions

##### `create_graph(use_sync_nodes: bool = False)`

Creates and compiles the modular MERR construction graph. It dynamically imports either asynchronous or synchronous node functions based on the `use_sync_nodes` flag, which is determined by the selected LLM (Hugging Face models run synchronously).

The function adds all nodes to the graph, sets the entry point, and defines the full control flow using edges and conditional routers.

##### Routing Functions

A series of small functions determine the next step in the graph based on the current `MERRState`.

* `route_by_processing_type`: The main router after setup. It directs the flow to the entry node of the selected pipeline (`MER`, `AU`, `audio`, `video`, or `image`).
* `route_after_emotion_filter`: After AU analysis, this routes to either save the results (for the `AU` pipeline) or proceed to the next step in the `MER` pipeline.
* `route_after_audio_generation`: After audio analysis, this routes to either save the results (for the `audio` pipeline) or proceed to the next step in the `MER` pipeline.
* `route_after_video_generation`: After video analysis, this routes to either save the results (for the `video` pipeline) or proceed to the next step in the `MER` pipeline.

### `mer_factory/prompts.py` - Prompt Templates

This module centralizes all prompt templates used for interacting with the LLMs.

#### Class: `PromptTemplates`

A class that stores and manages all prompt templates as static methods.

##### `describe_facial_expression()`

Returns a prompt instructing an LLM to act as a Facial Action Coding System (FACS) expert and describe a facial expression based on a list of Action Units.

##### `describe_image(has_label: bool = False)`

Returns a prompt for analyzing an image.
* If `has_label` is `True`, the prompt includes the ground truth label and asks for an objective description of the main subject, clothing, and scene.
* If `False`, it asks for the same objective description without a label.

##### `analyze_audio(has_label: bool = False)`

Returns a prompt for analyzing an audio file. It asks for two tasks: speech transcription and a description of audio characteristics (tone, pitch, etc.).
* If `has_label` is `True`, the ground truth label is included in the prompt.

##### `describe_video(has_label: bool = False)`

Returns a prompt for describing the content of a video, focusing on objective visual elements, people, and actions.
* If `has_label` is `True`, the ground truth label is included in the prompt.

##### `synthesize_summary(has_label: bool = False)`

Returns a prompt for a final analysis that synthesizes all gathered multimodal clues.
* If `has_label` is `True`, the prompt asks the LLM to act as a psychologist and build a rationale explaining why the ground truth label is correct based on the evidence.
* If `has_label` is `False`, the prompt asks the LLM to infer the subject's emotional state and narrate their emotional journey based on the clues.

### `mer_factory/models/__init__.py` - LLM Factory

This module provides a factory class for initializing the correct LLM based on provided CLI arguments.

#### Class: `LLMModels`

A class that selects and initializes the appropriate LLM (e.g., ChatGPT, Ollama, Hugging Face, Gemini) using a dictionary dispatch pattern.

##### `__init__(self, api_key: str, ollama_text_model_name: str, ollama_vision_model_name: str, chatgpt_model_name: str, huggingface_model_id: str, verbose: bool)`

Initializes the model instance based on the provided arguments. It iterates through a list of supported model types and instantiates the first one for which the necessary conditions are met.

* **Parameters:**
    * `api_key` (str, optional): A generic API key used for services like OpenAI or Gemini.
    * `ollama_text_model_name` (str, optional): Name of the Ollama text model.
    * `ollama_vision_model_name` (str, optional): Name of the Ollama vision model.
    * `chatgpt_model_name` (str, optional): Name of the ChatGPT model (e.g., 'gpt-4o').
    * `huggingface_model_id` (str, optional): ID of the Hugging Face model.
    * `verbose` (bool): Whether to print verbose logs. Defaults to `True`.
* **Raises:**
    * `ValueError`: If no model can be initialized because the required arguments (e.g., model name and/or API key) are not provided.

### `mer_factory/nodes/async_nodes.py` & `mer_factory/nodes/sync_nodes.py` - Asynchronous vs. Synchronous Execution

This section outlines the key differences between the asynchronous and synchronous node implementations. The choice between them is determined by the selected LLM, as some models (like Hugging Face) require synchronous execution.

#### Key Differences ⚖️

The primary difference between `sync_nodes.py` and `async_nodes.py` lies in their execution model. `sync_nodes.py` contains functions designed for **synchronous (sequential) execution**, which is necessary for models like Hugging Face that may not support asynchronous operations. In contrast, `async_nodes.py` is built for **asynchronous (concurrent) execution**, which improves performance by allowing the program to work on other tasks while waiting for I/O operations to complete.

#### Specific Distinctions

* **Function Definitions**: All functions in `async_nodes.py` are defined with the `async def` syntax, and `await` is used for non-blocking calls. `sync_nodes.py` uses standard `def` for all its functions.

* **Handling Blocking Operations**: `async_nodes.py` uses `asyncio.to_thread()` to run blocking I/O operations, like saving files or initializing the `FacialAnalyzer` class, in a separate thread. This prevents these operations from blocking the entire event loop. The synchronous version performs these tasks directly.

* **Concurrent API Calls**: In `async_nodes.py`, the `run_image_analysis` function executes two independent API calls concurrently using `asyncio.gather` to get the LLM's description of the Action Units (AUs) and the overall visual description. The synchronous version performs these calls sequentially.

---

## API vs. Local Model Implementations

The application supports various models that can be categorized as either API-based (requiring a remote service) or local (running on the user's machine).

### Gemini Model (API-based)

The `GeminiModel` class handles all interactions with the Google Gemini API.

* **Model**: It uses a single, multimodal model, `gemini-2.0-flash-lite`, for all tasks, including text, image, audio, and video analysis. The same model instance is used for both general and vision-specific tasks.
* **Initialization**: The model is initialized using a Google API key.
* **Functionality**:
    * It can analyze text, images, audio, and video by encoding the media file into base64 and sending it with a prompt.
    * All interactions with the API are asynchronous.

### Ollama Model (Hybrid Local/API)

The `OllamaModel` class provides a hybrid approach, using local Ollama models for text and vision tasks while leveraging a local Hugging Face model for audio processing.

* **Model**:
    * It can be configured with separate Ollama models for text and vision tasks. If only a vision model is provided, it is also used for text-based tasks.
    * For audio, it uses a Hugging Face pipeline, specifically `"openai/whisper-base"`.
* **Initialization**: It is initialized by providing names for the Ollama text and vision models. The Hugging Face audio model is initialized on a detected device (CUDA, MPS, or CPU).
* **Functionality**:
    * Text and image generation calls are asynchronous.
    * Audio analysis is performed synchronously using the Hugging Face pipeline.
    * Video analysis is explicitly not supported and will return an empty string.

### Hugging Face Models (Local)

The application supports a variety of specific, locally-run Hugging Face models through a dynamic loading system.

* **Model Registry**: A `HUGGINGFACE_MODEL_REGISTRY` contains a dictionary of supported models, mapping a model ID to its specific module and class name.
    * Examples include `"google/gemma-3n-E4B-it"`, `"Qwen/Qwen2-Audio-7B-Instruct"`, and `"zhifeixie/Audio-Reasoner"`.
* **Dynamic Loading**: The `get_hf_model_class` function provides "lazy loading" by only importing the necessary module for a given model when it is requested. This avoids dependency conflicts and unnecessary memory usage.
* **Error Handling**: If an unsupported model ID is requested, the system raises a `ValueError`.

---