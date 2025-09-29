# 👉🏻 MER-Factory 👈🏻
<p align="left">
        <a href="README_zh.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>
<br>

<p align="center">
  <a href="https://lum1104.github.io/MER-Factory/" target="_blank">📖 Documentation</a>
</p>

<p align="center"> <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Reasoning-red"> <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Recognition-red"> <img src="https://zenodo.org/badge/1007639998.svg" alt="DOI"> </p>

<p align="center">
  <a href="https://lum1104.github.io/MER-Factory/">
    <img src="docs/assets/logo.svg" width="700">
  </a>
 </p>

> [!IMPORTANT]
> ✍️ **Challenge:** Multimodal Affect Computing isn't just one step-it's the entire fragmented pipeline. The journey from raw files to a trained model is a gauntlet of tedious data preprocessing, slow and inconsistent manual annotation, and complex model training setups.
> 
> 🏭 **MER-Factory:** Unifies this entire workflow into one seamless factory. We automate the heavy lifting of preprocessing and annotation to generate high-quality, reason-augmented datasets, and then bridge the gap directly to model training.
> 
> 🚀 **Stop juggling different tools:** Let our factory handle the pipeline, so you can focus on what matters: your research.

<!-- <p align="center">
  <a href="https://lum1104.github.io/MER-Factory/">
    <img src="https://svg-banners.vercel.app/api?type=origin&text1=MER-Factory%20🧰&text2=✨%20Factory%20for%20Multimodal%20Emotion%20Recognition%20Reasoning%20(MERR)%20datasets&width=800&height=200" alt="MER-Factory Banner">
  </a>
</p> -->

## 🚀 Project Roadmap

MER-Factory is under active development with new features being added regularly - check our [roadmap](https://github.com/Lum1104/MER-Factory/wiki) and welcome contributions!

<div style="text-align: center;">
  <img src="docs/assets/framework.svg" style="border: none; width: 100%; max-width: 1000px;">
</div>

## Table of Contents

- [Pipeline Structure](#pipeline-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Command Structure](#basic-command-structure)
  - [Examples](#examples)
  - [Hugging Face Client-Server Setup](#hugging-face-client-server-setup)
  - [Command Line Options](#command-line-options)
  - [Processing Types](#processing-types)
  - [Export the Dataset](#export-the-dataset)
  - [Evaluate the Results](#evaluate-the-results)
- [Model Support](#model-support)
  - [Model Recommendations](#model-recommendations)
- [Training](#training)
- [Citation](#citation)


## Pipeline Structure

<details>
<summary>Click here to expand/collapse</summary>

Remove for now, call the (print(app.get_graph().draw_mermaid())) graph.py to view

</details>

## Features

-   **Action Unit (AU) Pipeline**: Extracts facial Action Units (AUs) and translates them into descriptive natural language.
-   **Audio Analysis Pipeline**: Extracts audio, transcribes speech, and performs detailed tonal analysis.
-   **Video Analysis Pipeline**: Generates comprehensive descriptions of video content and context.
-   **Image Analysis Pipeline**: Provides end-to-end emotion recognition for static images, complete with visual descriptions and emotional synthesis.
-   **Full MER Pipeline**: An end-to-end multimodal pipeline that identifies peak emotional moments, analyzes all modalities (visual, audio, facial), and synthesizes a holistic emotional reasoning summary.

Check out example outputs here:
-   [llava-llama3_llama3.2_merr_data.json](examples/llava-llama3_llama3.2_merr_data.json)
-   [gemini_merr.json](examples/gemini_merr.json)

## Installation

<p align="center">
  📚 Please visit <a href="https://lum1104.github.io/MER-Factory/" target="_blank">project documentation</a> for detailed installation and usage instructions.
</p>

A complete installation example for Linux environments (including Google Colab) can be found at:
- [`examples/MER_Factory.ipynb`](examples/MER_Factory.ipynb)

## Usage

### Basic Command Structure
```bash
python main.py [INPUT_PATH] [OUTPUT_DIR] [OPTIONS]
```

### Examples
```bash
# Show all supported args.
python main.py --help

# Full MER pipeline with Gemini (default)
python main.py path_to_video/ output/ --type MER --silent --threshold 0.8

# Using Sentiment Analysis task instead of MERR
python main.py path_to_video/ output/ --type MER --task "Sentiment Analysis" --silent

# Using ChatGPT models
python main.py path_to_video/ output/ --type MER --chatgpt-model gpt-4o --silent

# Using local Ollama models
python main.py path_to_video/ output/ --type MER --ollama-vision-model llava-llama3:latest --ollama-text-model llama3.2 --silent

# Using Hugging Face model
python main.py path_to_video/ output/ --type MER --huggingface-model google/gemma-3n-E4B-it --silent

# Process images instead of videos
python main.py ./images ./output --type MER
```

Note: Run `ollama pull llama3.2` etc, if Ollama model is needed. Ollama does not support video analysis for now.

### Hugging Face Client-Server Setup

When selecting a Hugging Face model with `--huggingface-model`, MER-Factory forwards all calls through a lightweight client that talks to a local/remote API server which actually hosts the HF model. This keeps your main environment clean and allows easy scaling.

1) Start the HF API Server (in a separate terminal):

```bash
# Example: serve Whisper base on port 7860
python -m mer_factory.models.hf_api_server --model_id openai/whisper-base --host 0.0.0.0 --port 7860
```

2) Run MER-Factory as usual and select the HF model by ID:

```bash
python main.py path_to_video/ output/ --type MER --huggingface-model openai/whisper-base --silent
```

### Dashboard for Data Curation and Hyperparameter Tuning

We provide an interactive dashboard webpage to facilitate data curation and hyperparameter tuning. The dashboard allows you to test different prompts, save and run configurations, and rate the generated data.

To launch the dashboard, use the following command:

```bash
python dashboard.py
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--type` | `-t` | Processing type (AU, audio, video, image, MER) | MER |
| `--task` | `-tk` | Analysis task type (MERR, Sentiment Analysis) | MERR |
| `--label-file` | `-l` | Path to a CSV file with 'name' and 'label' columns. Optional, for ground truth labels. | None |
| `--threshold` | `-th` | Emotion detection threshold (0.0-5.0) | 0.8 |
| `--peak_dis` | `-pd` | Steps between peak frame detection (min 8) | 15 |
| `--silent` | `-s` | Run with minimal output | False |
| `--cache` | `-ca` | Reuse existing audio/video/AU results from previous pipeline runs | False |
| `--concurrency` | `-c` | Concurrent files for async processing (min 1) | 4 |
| `--ollama-vision-model` | `-ovm` | Ollama vision model name | None |
| `--ollama-text-model` | `-otm` | Ollama text model name | None |
| `--chatgpt-model` | `-cgm` | ChatGPT model name (e.g., gpt-4o) | None |
| `--huggingface-model` | `-hfm` | Hugging Face model ID | None |

### Processing Types

#### 1. Action Unit (AU) Extraction
Extracts facial Action Units and generates natural language descriptions:
```bash
python main.py video.mp4 output/ --type AU
```

#### 2. Audio Analysis
Extracts audio, transcribes speech, and analyzes tone:
```bash
python main.py video.mp4 output/ --type audio
```

#### 3. Video Analysis
Generates comprehensive video content descriptions:
```bash
python main.py video.mp4 output/ --type video
```

#### 4. Image Analysis
Runs the pipeline with image input:
```bash
python main.py ./images ./output --type image
# Note: Image files will automatically use image pipeline regardless of --type setting
```

#### 5. Full MER Pipeline (Default)
Runs the complete multimodal emotion recognition pipeline:
```bash
python main.py video.mp4 output/ --type MER
# or simply:
python main.py video.mp4 output/
```

### Task Types

The `--task` option allows you to choose between different analysis tasks:

#### 1. Emotion Recognition (Default)
Performs detailed emotion analysis with granular emotion categories:
```bash
python main.py video.mp4 output/ --task "MERR"
# or simply omit the --task option since it's the default
python main.py video.mp4 output/
```

#### 2. Sentiment Analysis
Performs sentiment-focused analysis (positive, negative, neutral):
```bash
python main.py video.mp4 output/ --task "Sentiment Analysis"
```

### Export the Dataset

To export datasets for curation or training, use the following commands:

#### For Dataset Curation
```bash
python export.py --output_folder "{output_folder}" --file_type {file_type.lower()} --export_path "{export_path}" --export_csv
```

#### For Training
```bash
python export.py --input_csv path/to/csv_file.csv --export_format sharegpt
```

### Evaluate the Results

MER-Factory includes a comprehensive reference-free evaluation toolkit to assess the quality of generated annotations without human ratings.

#### Basic Evaluation
```bash
# Evaluate all samples in output directory
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv
```

#### Advanced Evaluation Options
```bash
# Run with verbose output to see detailed failure reasons
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv --verbose

# Skip writing per-sample evaluation files
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv --no-write-per-sample
```

#### Evaluation Metrics
The evaluation toolkit provides multiple quality metrics:

- **🖼️ CLIP Image Score**: Visual grounding between images and descriptions
- **🔊 CLAP Audio Score**: Audio-text alignment using LAION-CLAP
- **😊 AU F1 Score**: Facial expression accuracy vs OpenFace AUs
- **🔗 NLI Consistency**: Logical consistency across modalities
- **🎙️ ASR WER**: Speech recognition quality vs Whisper baseline
- **📝 Text Quality**: Distinctness, repetition, and readability metrics
- **🎯 Composite Score**: Overall quality (0-100) combining all metrics

#### Evaluation Output
- **Per-sample**: `evaluation.json` files in each sample directory
- **Dataset-level**: `evaluation_summary.csv` with rankings and statistics
- **Console**: Beautiful progress bars and top-performing samples table

For detailed evaluation documentation, see [`tools/evaluate/README.md`](tools/evaluate/README.md).

## Model Support

The tool supports four types of models:

1. **Google Gemini** (default): Requires `GOOGLE_API_KEY` in `.env`
2. **OpenAI ChatGPT**: Requires `OPENAI_API_KEY` in `.env`, specify with `--chatgpt-model`
3. **Ollama**: Local models, specify with `--ollama-vision-model` and `--ollama-text-model`
4. **Hugging Face**: Currently supports multimodal models like `google/gemma-3n-E4B-it`

**Note**: If using Hugging Face models, concurrency is automatically set to 1 for synchronous processing.

### Model Recommendations

#### When to Use Ollama
**Recommended for**: Image analysis, Action Unit analysis, text processing, and simple audio transcription tasks.

**Benefits**:
- ✅ **Async support**: Ollama supports asynchronous calling, making it ideal for processing large datasets efficiently
- ✅ **Local processing**: No API costs or rate limits
- ✅ **Wide model selection**: Visit [ollama.com](https://ollama.com/) to explore available models
- ✅ **Privacy**: All processing happens locally

**Example usage**:
```bash
# Process images with Ollama
python main.py ./images ./output --type image --ollama-vision-model llava-llama3:latest --ollama-text-model llama3.2 --silent

# AU extraction with Ollama
python main.py video.mp4 output/ --type AU --ollama-text-model llama3.2 --silent
```

#### When to Use ChatGPT/Gemini
**Recommended for**: Advanced video analysis, complex multimodal reasoning, and high-quality content generation.

**Benefits**:
- ✅ **State-of-the-art performance**: Latest GPT-4o and Gemini models offer superior reasoning capabilities
- ✅ **Advanced video understanding**: Better support for complex video analysis and temporal reasoning
- ✅ **High-quality outputs**: More nuanced and detailed emotion recognition and reasoning
- ✅ **Robust multimodal integration**: Excellent performance across text, image, and video modalities

**Example usage**:
```bash
python main.py video.mp4 output/ --type MER --chatgpt-model gpt-4o --silent

python main.py video.mp4 output/ --type MER --silent
```

**Trade-offs**: API costs and rate limits, but typically provides the highest quality results for complex emotion reasoning tasks.

#### When to Use Hugging Face Models
**Recommended for**: When you need the latest state-of-the-art models or specific features not available in Ollama.

**Custom Model Integration**:
If you want to use the latest HF models or features that Ollama doesn't support:

1. **Option 1 - Implement yourself**: Navigate to `mer_factory/models/hf_models/__init__.py` to register your own model and implement the needed functions following our existing patterns.

2. **Option 2 - Request support**: Open an issue on our repository to let us know which model you'd like us to support, and we'll consider adding it.

**Current supported models**: `google/gemma-3n-E4B-it` and others listed in the HF models directory.

## Training

This training guide will walk you through the complete end-to-end process from **Data Analysis/Annotation** to **Launching Model Training**. The process is divided into two main stages:

1.  **Stage One: Automated Data Preparation**: Use the `train.sh` script to convert the analysis output from MER-Factory into the standard dataset format required by the training framework with a single command, and automatically complete the registration.
2.  **Stage Two: Interactive Training Launch**: Start the LLaMA-Factory graphical user interface (Web UI), load the prepared dataset, and freely configure all training parameters.


### Prerequisites

Before you begin, please ensure that you have completed the following environmental preparations:

1.  **Initialize Submodules**
   
   This project uses Git Submodules to integrate LLaMA-Factory to ensure version consistency and reproducibility of the training environment.
   
   After cloning this repository, please run the following command to initialize and download the submodules:
   ```bash
   git submodule update --init --recursive
   ```

2.  **Install Dependencies**
   This project and the LLaMA-Factory submodule have their own separate dependency environments, which need to be installed individually:
   ```bash
   # 1. Install the main dependencies for MER-Factory
   pip install -r requirements.txt

   # 2. Install the dependencies for the LLaMA-Factory submodule
   pip install -r LLaMA-Factory/requirements.txt
   ```

### Stage One: Automated Data Preparation

After you have finished analyzing the raw data using **`main.py`**, you can use the `train.sh` script to prepare the dataset.

The core task of this script is to **automate all the tedious data preparation work**. It reads the analysis results from MER-Factory, converts them into the ShareGPT format required by LLaMA-Factory, and automatically registers the dataset within LLaMA-Factory.

#### Usage Example

To ensure the traceability and consistency of experiments, we recommend naming your dataset using the following format:

`RawDataset_AnalysisModel_TaskType`

Process data for an **MER** task and name the dataset according to the convention:
```bash
# Assuming the llava and llama3.2 analysis models were used
bash train.sh --file_type "image" --dataset_name "mer2025_llava_llama3.2_MER"
```

Process data for an **audio** task and name the dataset according to the convention:
```bash
# Assuming the gemini api model was used
bash train.sh --file_type "audio" --dataset_name "mer2025_gemini_audio"
```

Process data for a **video** task and name the dataset according to the convention:
```bash
# Assuming the gemini api model was used
bash train.sh --file_type "video" --dataset_name "mer2025_gemini_video"
```

Process data for an **image** task and name the dataset according to the convention:
```bash
# Assuming the chatgpt gpt-4o model was used
bash train.sh --file_type "mer" --dataset_name "mer2025_gpt-4o_image"
```

After the script runs successfully, your dataset (e.g., mer2025_llava_llama3.2_MER) will be ready and registered in LLaMA-Factory's dataset_info, making it directly available for use in the next stage.

### Stage Two: Launch Training (Start LLaMA-Factory Web UI)

Once your dataset is ready, you can launch the LLaMA-Factory graphical interface to configure and start your training task.

1. **Navigate to the LLaMA-Factory Directory**
   
   ```bash
   cd LLaMA-Factory
   ```
2. **Start the Web UI**
   
   ```bash
   llamafactory-cli webui
   ```
3. **Configure and Train in the Web UI**



## Citation

If you find MER-Factory useful in your research or project, please consider giving us a ⭐! Your support helps us grow and continue improving.

Additionally, if you use MER-Factory in your work, please consider cite us using the following BibTeX entries:

```bibtex
@software{Lin_MER-Factory_2025,
  author = {Lin, Yuxiang and Zheng, Shunchao},
  doi = {10.5281/zenodo.15847351},
  license = {MIT},
  month = {7},
  title = {{MER-Factory}},
  url = {https://github.com/Lum1104/MER-Factory},
  version = {0.1.0},
  year = {2025}
}

@inproceedings{NEURIPS2024_c7f43ada,
  author = {Cheng, Zebang and Cheng, Zhi-Qi and He, Jun-Yan and Wang, Kai and Lin, Yuxiang and Lian, Zheng and Peng, Xiaojiang and Hauptmann, Alexander},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {110805--110853},
  publisher = {Curran Associates, Inc.},
  title = {Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/c7f43ada17acc234f568dc66da527418-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
```
