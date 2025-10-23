---
layout: default
title: Getting Started
description: Quick start guide for MER-Factory installation and basic usage
lang: en
---

# Getting Started with MER-Factory

Get up and running with MER-Factory in just a few minutes. This guide will walk you through the installation process and your first emotion recognition pipeline.

## System Overview

<div style="text-align: center;">
  <img src="assets/framework.svg" style="border: none; width: 100%; max-width: 1000px;">
</div>

## Prerequisites

Before installing MER-Factory, ensure you have the following dependencies installed on your system:

### 1. FFmpeg Installation

FFmpeg is required for video and audio processing.

<div class="feature-grid">
  <div class="feature-card">
    <h3><i class="fab fa-apple"></i> macOS</h3>
    <pre><code>brew install ffmpeg</code></pre>
  </div>
  
  <div class="feature-card">
    <h3><i class="fab fa-ubuntu"></i> Ubuntu/Debian</h3>
    <pre><code>sudo apt update && sudo apt install ffmpeg</code></pre>
  </div>
  
  <div class="feature-card">
    <h3><i class="fab fa-windows"></i> Windows</h3>
    <p>Download from <a href="https://ffmpeg.org/download.html">ffmpeg.org</a></p>
  </div>
</div>

**Verify installation:**
```bash
ffmpeg -version
ffprobe -version
```

### 2. OpenFace Installation
Facial Action Unit extraction requires [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki).

#### Mac/Linux Installation Method

[Reference Code](https://github.com/Lum1104/MER-Factory/blob/main/examples/MER_Factory.ipynb)

<div class="alert alert-info">
<strong>Note:</strong> After building OpenFace, please note down the path to the <code>FeatureExtraction</code> executable (usually in <code>build/bin/FeatureExtraction</code>). You will need it during configuration.
</div>

#### Windows Installation and Configuration Guide
1. Download the [pre-compiled version](https://github.com/TadasBaltrusaitis/OpenFace/releases) for Windows.
2. Run `download_models.ps1` in PowerShell. If you cannot access Dropbox, follow the [model download documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download) to download the models and place them in the `model/patch_experts` path.
3. Enter the **absolute path** of the `FeatureExtraction.exe` file into the `.env` file.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Lum1104/MER-Factory.git
cd MER-Factory
```

### 2. Set Up Python Environment

```bash
# Create a new conda environment
conda create -n mer-factory python=3.12
conda activate mer-factory

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env
```

Edit the `.env` file with your settings:

```env
# API Keys (optional - choose based on your preferred models)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# OpenFace Configuration (required for AU and MER pipelines)
OPENFACE_EXECUTABLE=/absolute/path/to/OpenFace/build/bin/FeatureExtraction

# Optional: Ollama configuration for local models
# OLLAMA_HOST=http://localhost:11434
```

<div class="alert alert-warning">
<strong>Important:</strong> The <code>OPENFACE_EXECUTABLE</code> path must be absolute and point to the actual executable file.
</div>

## Your First Pipeline

Let's run your first emotion recognition pipeline!

### 1. Prepare Your Media

Create a test directory with a video file:

```bash
mkdir test_input
# Copy your video file to test_input/your_video.mp4
```

### 2. Run MER Pipeline

```bash
# Basic MER pipeline with default Gemini model
python main.py test_input/ output/ --type MER --silent

# With threshold adjustment
python main.py test_input/ output/ --type MER --threshold 0.8 --silent
```

### 3. Check Results

```bash
# View generated files
ls output/{sample_id}/
# your_video_merr_data.json - Contains complete analysis
# your_video_au_data.csv - Facial Action Units data
# your_video.wav - Extracted audio
# your_video_peak_frame.jpg - Key emotional moment
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

## Model Options

MER-Factory supports multiple AI models. Choose based on your needs:

### Google Gemini (Default)
```bash
python main.py input/ output/ --type MER
```
- **Best for:** High-quality multimodal analysis
- **Requires:** `GOOGLE_API_KEY` in `.env`

### OpenAI ChatGPT
```bash
python main.py input/ output/ --type MER --chatgpt-model gpt-4o
```
- **Best for:** Advanced reasoning and video analysis
- **Requires:** `OPENAI_API_KEY` in `.env`

### Ollama (Local Models)
```bash
# First, pull the models
ollama pull llava-llama3:latest
ollama pull llama3.2

# Run with Ollama
python main.py input/ output/ --type MER \
  --ollama-vision-model llava-llama3:latest \
  --ollama-text-model llama3.2
```
- **Best for:** Privacy, no API costs, async processing
- **Requires:** Local Ollama installation

### Hugging Face Models
```bash
python main.py input/ output/ --type MER --huggingface-model google/gemma-3n-E4B-it
```
- **Best for:** Latest research models, custom implementations
- **Note:** Automatic single-threaded processing

## Pipeline Types

### Quick Pipeline Comparison

| Pipeline | Input | Output | Use Case |
|----------|-------|---------|----------|
| **MER** | Video/Image | Complete emotion analysis | Full multimodal datasets |
| **AU** | Video | Facial Action Units | Facial expression research |
| **Audio** | Video | Speech + tone analysis | Audio emotion recognition |
| **Video** | Video | Visual description | Video understanding |
| **Image** | Images | Image emotion analysis | Static emotion recognition |

### Example Commands

```bash
# Action Unit extraction only
python main.py video.mp4 output/ --type AU

# Audio analysis only  
python main.py video.mp4 output/ --type audio

# Video description only
python main.py video.mp4 output/ --type video

# Image analysis (auto-detected for image inputs)
python main.py ./images/ output/ --type image

# Full MER with custom settings
python main.py videos/ output/ \
  --type MER \
  --threshold 0.9 \
  --peak-dis 20 \
  --concurrency 8 \
  --silent
```

## Testing Your Installation

Run the built-in tests to verify everything is working:

```bash
# Test FFmpeg integration
python test/test_ffmpeg.py your_video.mp4 test_output/

# Test OpenFace integration  
python test/test_openface.py your_video.mp4 test_output/
```

## Common Issues & Solutions

### FFmpeg Not Found
**Symptom:** `FileNotFoundError` related to `ffmpeg`

**Solution:** 
1. Verify FFmpeg is installed: `ffmpeg -version`
2. Check if it's in your PATH
3. On Windows, add FFmpeg to system PATH

### OpenFace Executable Not Found
**Symptom:** Cannot find FeatureExtraction executable

**Solution:**
1. Verify the path in `.env` is absolute
2. Check file permissions: `chmod +x FeatureExtraction`
3. Test manually: `/path/to/FeatureExtraction -help`

### API Key Errors
**Symptom:** `401 Unauthorized` errors

**Solution:**
1. Verify API keys are correct in `.env`
2. Check for extra spaces or characters
3. Ensure billing is enabled for your API account

### Memory Issues
**Symptom:** Out of memory errors with large files

**Solution:**
1. Reduce concurrency: `--concurrency 1`
2. Use smaller video files for testing
3. Close other memory-intensive applications

## Next Steps

Now that you have MER-Factory running, explore these advanced features:

- **[API Reference](/MER-Factory/api-reference)** - Detailed function documentation
- **[Examples](/MER-Factory/examples)** - Real-world usage examples
- **[Technical Documentation](/MER-Factory/technical-docs)** - System architecture details

## Need Help?

- 🐛 Report issues on [GitHub Issues](https://github.com/Lum1104/MER-Factory/issues)
- 💬 Join discussions on [GitHub Discussions](https://github.com/Lum1104/MER-Factory/discussions)
- 📖 Read the [Technical Documentation](/MER-Factory/technical-docs) for deeper understanding

<style>
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.feature-card {
  padding: 1.5rem;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  text-align: center;
}

.feature-card h3 {
  margin: 0 0 1rem 0;
  color: var(--secondary-color);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.feature-card pre {
  margin: 0;
  text-align: left;
}
</style>
