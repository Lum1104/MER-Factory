---
layout: default
title: MER-Factory - Multimodal Emotion Recognition Factory
description: Your automated factory for constructing Multimodal Emotion Recognition and Reasoning (MERR) datasets
lang: en
---

<div class="hero-section">
  <h1 class="hero-title">👉🏻 MER-Factory 👈🏻</h1>
  <p class="hero-subtitle">Your automated factory for constructing Multimodal Emotion Recognition and Reasoning (MERR) datasets</p>
  
  <div class="badges">
    <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Reasoning-red" alt="MERR">
    <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Recognition-red" alt="MER">
    <img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
    <img src="https://zenodo.org/badge/1007639998.svg" alt="DOI">
  </div>

  <img src="assets/logo.svg" alt="MER-Factory Logo" class="hero-image">
</div>

## 🚀 Project Roadmap

MER-Factory is under active development with new features being added regularly - check our [roadmap](https://github.com/Lum1104/MER-Factory/wiki) and welcome contributions!

## Quick Overview

MER-Factory is a **Python-based, open-source framework** designed for the Affective Computing community. It automates the creation of unified datasets for training Multimodal Large Language Models (MLLMs) by extracting multimodal features and leveraging LLMs to generate detailed analyses and emotional reasoning summaries.

### 🚀 Key Features

- **Multi-Pipeline Architecture**: Support for AU, Audio, Video, Image, and full MER processing.
- **Flexible Analysis Tasks**: Choose between MERR and Sentiment Analysis.
- **Flexible Model Integration**: Works with OpenAI, Google Gemini, Ollama, and Hugging Face models.
- **Scalable Processing**: Async/concurrent processing for large datasets.
- **Scientific Foundation**: Based on Facial Action Coding System (FACS) and latest research.
- **Easy CLI Interface**: Simple command-line usage with comprehensive options.
- **Interactive Tools**: Web-based dashboard for data curation and configuration management.

### 📋 Processing Types

| Pipeline | Description | Use Case |
|---|---|---|
| **AU** | Facial Action Unit extraction and description. | Facial expression analysis. |
| **Audio** | Speech transcription and tonal analysis. | Audio emotion analysis. |
| **Video** | Comprehensive video content description. | Video emotion analysis. |
| **Image** | Static image emotion recognition. | Image-based emotion analysis. |
| **MER** | Complete multimodal pipeline. | Full emotion reasoning datasets. |

### 🎯 Analysis Task Types

The `--task` argument allows you to specify the analysis goal.

| Task | `--task` argument | Description |
|---|---|---|
| **MERR** | `"MERR"` | (Default) Performs detailed analysis with MER. |
| **Sentiment Analysis** | `"Sentiment Analysis"` | Performs sentiment-focused analysis (positive, negative, neutral). |

### 📖 Example Outputs

Check out real examples of what MER-Factory produces:
- [Gemini Model Output](https://github.com/Lum1104/MER-Factory/blob/main/examples/gemini_merr.json)
- [LLaVA Model Output](https://github.com/Lum1104/MER-Factory/blob/main/examples/llava-llama3_llama3.2_merr_data.json)

## Architecture Overview

- **CLI Framework**: Utilizes Typer for a robust and user-friendly command-line interface.
- **Workflow Management**: Employs LangGraph to enable stateful and dynamic processing pipelines.
- **Facial Analysis**: Integrates OpenFace for precise Facial Action Units extraction.
- **Media Processing**: Leverages FFmpeg for advanced audio and video manipulation tasks.
- **AI Integration**: Features a pluggable architecture supporting multiple LLM providers.
- **Concurrency**: Implements Asyncio for efficient and scalable parallel processing.

## Getting Started

Ready to dive in? Here's what you need to know:

1.  **[Prerequisites](/MER-Factory/getting-started#prerequisites)** - Install FFmpeg and OpenFace
2.  **[Installation Guide](/MER-Factory/getting-started#installation)** - Set up MER-Factory
3.  **[Basic Usage](/MER-Factory/getting-started#your-first-pipeline)** - Your first emotion recognition pipeline
4.  **[Model Configuration](/MER-Factory/getting-started#model-options)** - Choose and configure your AI models
5.  **[Advanced Features](/MER-Factory/getting-started#next-steps)** - Explore all capabilities

## Community & Support

- 📚 **[Technical Documentation](/MER-Factory/technical-docs)** - Deep dive into system architecture
- 🔧 **[API Reference](/MER-Factory/api-reference)** - Complete function and class documentation
- 💡 **[Examples](/MER-Factory/examples)** - Real-world usage examples and tutorials
- 🐛 **Issues & Bug Reports** - [GitHub Issues](https://github.com/Lum1104/MER-Factory/issues)
- 💬 **Discussions** - [GitHub Discussions](https://github.com/Lum1104/MER-Factory/discussions)

*Advancing together with the Affective Computing community.*

<style>
.hero-section {
  text-align: center;
  margin: 2rem 0;
  padding: 2rem;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 10px;
}

.hero-title {
  font-size: 2.5rem;
  margin-bottom: 0.2rem;
  color: #2c3e50;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: #7f8c8d;
  margin-bottom: 0.2rem;
}

.badges {
  margin: 0.3rem 0;
  display: inline-block;
  text-align: center;
}

.badges img {
  height: 28px;
  margin: 0.4rem;
  vertical-align: middle;
  border: none;
  background: none;
  box-shadow: none;
}

.hero-image {
  max-width: 900px;
  width: 60%;
  height: auto;
  margin: 0.5rem auto;
  display: block;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .hero-image {
    width: 80%;
    max-width: 600px;
  }
  
  .badges {
    display: block;
    text-align: center;
  }
}

</style>
