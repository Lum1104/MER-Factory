---
layout: default
title: Tools
description: Interactive tools and utilities for MER-Factory
lang: en
---

# 🛠️ Tools

MER-Factory provides interactive tools to help you manage your data and configure processing pipelines.

## Tune Dashboard

The **Tune Dashboard** is a web-based interface that provides two main functionalities:

### 📊 Data Curation
- **CSV Upload & Preview**: Upload your dataset CSV files and preview them interactively
- **Media Preview**: View images and videos directly in the browser
- **Quality Rating**: Rate data samples with a 5-star rating system
- **Inline Editing**: Edit dataset fields directly in the interface
- **Export**: Export your curated data back to CSV format

### ⚙️ Prompt Configuration & CLI running
- **Prompt Editing**: Load and modify the default prompts used by MER-Factory
- **Visual Configuration**: Configure all MER-Factory parameters through a user-friendly interface
- **Model Selection**: Choose between different AI providers (Gemini, ChatGPT, Ollama, HuggingFace)
- **Command Generation**: Automatically generate the complete command line for your configuration
- **Download Prompts**: Export your customized prompts as `prompts_v{i}.json`
- **CLI Tool**: Run the CLI tool `main.py` directly to tune the hyper-parameters. Then export and load the constructed dataset directly into the `Data Curation` to rate and curate.

## Access the Dashboard

*Note: The preview functionality for videos/images is for demonstration purposes only. Please clone the repository locally run `dashboard.py` to use.*

<div class="tool-access">
  <a href="tune-dashboard.html" class="btn btn-primary">
    🚀 Open Tune Dashboard
  </a>
</div>

## Features Overview

### Data Management
- **Drag & Drop Upload**: Simply drag your CSV files to get started
- **Pagination**: Navigate through large datasets easily
- **Media Integration**: Preview videos and images with customizable path prefixes
- **Real-time Editing**: Changes are saved automatically as you edit

### Configuration Management
- **Prompt Templates**: Edit system prompts, user prompts, and task-specific templates
- **Parameter Tuning**: Adjust thresholds, concurrency, and processing options
- **Model Configuration**: Set up API keys and model parameters for different providers
- **Export Ready**: Generate production-ready command lines

### User Experience
- **Modern UI**: Clean, dark-themed interface built with Tailwind CSS
- **Responsive Design**: Works on desktop and mobile devices
- **Keyboard Shortcuts**: Efficient navigation and editing
- **Toast Notifications**: Clear feedback for all actions

## Getting Started with the Dashboard

1. **Open the Dashboard**: Click the link above to access the Tune Dashboard
2. **Upload Your Data**: Switch to the "Data Curation" tab and upload a CSV file
3. **Review and Edit**: Use the interface to review, rate, and edit your data
4. **Configure Processing**: Switch to "Prompt & Run" tab to set up your processing pipeline
5. **Generate Command**: Click "Generate Command" to get your ready-to-run CLI command
6. **Download Assets**: Export your edited prompts and run the generated command

## Technical Requirements

- **Browser**: Modern web browser with JavaScript enabled
- **File Formats**: CSV files for data, JSON files for prompts
- **Media Support**: Common image (JPG, PNG) and video (MP4, AVI) formats

---

*The Tune Dashboard is designed to streamline your MER-Factory workflow, making data curation and configuration management intuitive and efficient.*
