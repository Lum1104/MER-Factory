# 👉🏻 MER-Factory 👈🏻

<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>
<br>

<p align="center">
  <strong>您的多模态情感识别与推理（MERR）数据集自动化工厂。</strong>
</p>
<p align="center">
  <a href="https://lum1104.github.io/MER-Factory/" target="_blank">📖 项目文档</a>
</p>

<p align="center"> <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Reasoning-red"> <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Recognition-red"> <img src="https://zenodo.org/badge/1007639998.svg" alt="DOI"> </p>

<p align="center">
  <a href="https://lum1104.github.io/MER-Factory/zh/">
    <img src="docs/assets/logo.svg" width="700">
  </a>
 </p>

<!-- <p align="center">
  <a href="https://lum1104.github.io/MER-Factory/">
    <img src="https://svg-banners.vercel.app/api?type=origin&text1=MER-Factory%20🧰&text2=✨%20多模态情绪识别推理%20(MERR)%20数据集工厂&width=800&height=200" alt="MER-Factory Banner">
  </a>
</p> -->

## 🚀 项目路线图

MER-Factory 正在积极开发中，新功能会定期添加 - 查看我们的[路线图](https://github.com/Lum1104/MER-Factory/wiki)，欢迎贡献！

<div style="text-align: center;">
  <img src="docs/assets/framework.svg" style="border: none; width: 100%; max-width: 1000px;">
</div>

## 目录

- [Pipeline 结构](#pipeline-结构)
- [特性](#特性)
- [安装](#安装)
- [使用方法](#使用方法)
  - [基本命令结构](#基本命令结构)
  - [示例](#示例)
  - [命令行选项](#命令行选项)
  - [处理类型](#处理类型)
- [模型支持](#模型支持)
  - [模型推荐](#模型推荐)
- [引用](#引用)

## Pipeline 结构

<details>
<summary>点击展开/折叠</summary>

```mermaid
graph TD;
        __start__([<p>__start__</p>]):::first
        setup_paths(setup_paths)
        handle_error(handle_error)
        run_au_extraction(run_au_extraction)
        save_au_results(save_au_results)
        generate_audio_description(generate_audio_description)
        save_audio_results(save_audio_results)
        generate_video_description(generate_video_description)
        save_video_results(save_video_results)
        extract_full_features(extract_full_features)
        filter_by_emotion(filter_by_emotion)
        find_peak_frame(find_peak_frame)
        generate_peak_frame_visual_description(generate_peak_frame_visual_description)
        generate_peak_frame_au_description(generate_peak_frame_au_description)
        synthesize_summary(synthesize_summary)
        save_mer_results(save_mer_results)
        run_image_analysis(run_image_analysis)
        synthesize_image_summary(synthesize_image_summary)
        save_image_results(save_image_results)
        __end__([<p>__end__</p>]):::last
        __start__ --> setup_paths;
        extract_full_features --> filter_by_emotion;
        filter_by_emotion -.-> find_peak_frame;
        filter_by_emotion -.-> handle_error;
        filter_by_emotion -.-> save_au_results;
        find_peak_frame --> generate_audio_description;
        generate_audio_description -.-> generate_video_description;
        generate_audio_description -.-> handle_error;
        generate_audio_description -.-> save_audio_results;
        generate_peak_frame_au_description --> synthesize_summary;
        generate_peak_frame_visual_description --> generate_peak_frame_au_description;
        generate_video_description -.-> generate_peak_frame_visual_description;
        generate_video_description -.-> handle_error;
        generate_video_description -.-> save_video_results;
        run_au_extraction --> filter_by_emotion;
        run_image_analysis --> synthesize_image_summary;
        setup_paths -. &nbsp;full_pipeline&nbsp; .-> extract_full_features;
        setup_paths -. &nbsp;audio_pipeline&nbsp; .-> generate_audio_description;
        setup_paths -. &nbsp;video_pipeline&nbsp; .-> generate_video_description;
        setup_paths -.-> handle_error;
        setup_paths -. &nbsp;au_pipeline&nbsp; .-> run_au_extraction;
        setup_paths -. &nbsp;image_pipeline&nbsp; .-> run_image_analysis;
        synthesize_image_summary --> save_image_results;
        synthesize_summary --> save_mer_results;
        handle_error --> __end__;
        save_au_results --> __end__;
        save_audio_results --> __end__;
        save_image_results --> __end__;
        save_mer_results --> __end__;
        save_video_results --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

</details>

## 特性

-   **动作单元（AU）处理流程**：提取面部动作单元（AUs），并将其翻译成描述性的自然语言。
-   **音频分析处理流程**：提取音频，转录语音，并进行详细的音调分析。
-   **视频分析处理流程**：生成视频内容和上下文的全面描述。
-   **图像分析处理流程**：为静态图像提供端到端的情感识别，包括视觉描述和情感合成。
-   **完整 MER 处理流程**：端到端的多模态处理流程，识别情感峰值时刻，分析所有模态（视觉、音频、面部），并合成一个整体的情感推理总结。

查看示例输出：
-   [llava-llama3_llama3.2_merr_data.json](examples/llava-llama3_llama3.2_merr_data.json)
-   [gemini_merr.json](examples/gemini_merr.json)

## 安装

<p align="center">
  📚 请访问 <a href="https://lum1104.github.io/MER-Factory/zh/" target="_blank">项目文档</a> 查看详细的安装和使用教程。
</p>

## 使用方法

### 基本命令结构
```bash
python main.py [输入路径] [输出目录] [选项]
```

### 示例
```bash
# 查看所有支持功能
python main.py --help

# 使用 Gemini（默认）运行完整 MER 处理流程
python main.py path_to_video/ output/ --type MER --silent --threshold 0.8

# 使用情感分析任务代替情感识别
python main.py path_to_video/ output/ --type MER --task "Sentiment Analysis" --silent

# 使用 ChatGPT 模型
python main.py path_to_video/ output/ --type MER --chatgpt-model gpt-4o --silent

# 使用本地 Ollama 模型
python main.py path_to_video/ output/ --type MER --ollama-vision-model llava-llama3:latest --ollama-text-model llama3.2 --silent

# 使用 Hugging Face 模型
python main.py path_to_video/ output/ --type MER --huggingface-model google/gemma-3n-E4B-it --silent

# 处理图像而不是视频
python main.py ./images ./output --type MER
```

注意：如果需要使用 Ollama 模型，请运行 `ollama pull llama3.2` 等命令预先下载模型。Ollama 目前不支持视频分析。

### 交互式仪表板

我们提供了一个交互式仪表板网页，用于简化数据整理和超参数调优过程。

通过该仪表板，您可以方便地选择不同类型的模型、不同的预训练模型、任务类别，也可以根据您的需求更换提示词模板进行测试，支持一键式生成命令，并在仪表板直接运行 MER-Factory 以生成数据集，还可以对生成的数据进行评分。

要启动仪表板，请使用以下命令：

```bash
python dashboard.py
```

### 命令行选项

| 选项 | 简写 | 描述 | 默认值 |
|--------|-------|-------------|---------|
| `--type` | `-t` | 处理类型（AU、audio、video、image、MER） | MER |
| `--task` | `-tk` | 分析任务类型（Emotion Recognition、Sentiment Analysis） | Emotion Recognition |
| `--label-file` | `-l` | CSV 文件路径，包含 'name' 和 'label' 列。可选，用于真实标签。 | None |
| `--threshold` | `-th` | 情感检测阈值（0.0-5.0） | 0.8 |
| `--peak_dis` | `-pd` | 情感峰值帧检测间隔（最小 8） | 15 |
| `--silent` | `-s` | 以最小输出运行 | False |
| `--cache` | `-ca` | 复用现有 音视频/ AU 分析结果 | False |
| `--concurrency` | `-c` | 异步处理文件数量（最小 1） | 4 |
| `--ollama-vision-model` | `-ovm` | Ollama 视觉模型名称 | None |
| `--ollama-text-model` | `-otm` | Ollama 文本模型名称 | None |
| `--chatgpt-model` | `-cgm` | ChatGPT 模型名称（例如 gpt-4o） | None |
| `--huggingface-model` | `-hfm` | Hugging Face 模型 ID | None |

### 处理类型

#### 1. 动作单元（AU）提取
提取面部动作单元并生成自然语言描述：
```bash
python main.py video.mp4 output/ --type AU
```

#### 2. 音频分析
提取音频，转录语音并分析音调：
```bash
python main.py video.mp4 output/ --type audio
```

#### 3. 视频分析
生成视频内容的全面描述：
```bash
python main.py video.mp4 output/ --type video
```

#### 4. 图像分析
使用图像输入运行处理流程：
```bash
python main.py ./images ./output --type image
# 注意：图像文件将自动使用图像处理流程，无论 --type 设置为何值
```

#### 5. 完整 MER 处理流程（默认）
运行完整的多模态情感识别处理流程：
```bash
python main.py video.mp4 output/ --type MER
# 或者简单地：
python main.py video.mp4 output/
```

### 任务类型

`--task` 选项允许您在不同的分析任务之间进行选择：

#### 1. 情感识别（默认）
进行详细的情感分析，使用精细的情感类别：
```bash
python main.py video.mp4 output/ --task "Emotion Recognition"
# 或者直接省略 --task 选项，因为这是默认值
python main.py video.mp4 output/
```

#### 2. 情感分析
进行基于情绪极性的分析（积极、消极、中性）：
```bash
python main.py video.mp4 output/ --task "Sentiment Analysis"
```


### 导出数据集

要导出数据集以进行整理或训练，请使用以下命令：

#### 用于数据集整理
```bash
python export.py --output_folder "{output_folder}" --file_type {file_type.lower()} --export_path "{export_path}" --export_csv
```

#### 用于训练
```bash
python export.py --input_csv path/to/csv_file.csv --export_format sharegpt
```

## 模型支持

该工具支持四种类型的模型：

1. **Google Gemini**（默认）：需要在 `.env` 中配置 `GOOGLE_API_KEY`
2. **OpenAI ChatGPT**：需要在 `.env` 中配置 `OPENAI_API_KEY`，通过 `--chatgpt-model` 指定
3. **Ollama**：本地模型，需要预安装 ollama，通过 `--ollama-vision-model` 和 `--ollama-text-model` 指定
4. **Hugging Face**：目前支持类似 `google/gemma-3n-E4B-it` 的多模态模型

**注意**：如果使用 Hugging Face 模型，会自动将并发设置为 1，以实现同步处理。

### 模型推荐

#### 何时使用 Ollama
**推荐用途**：图像分析、动作单元分析、文本处理以及简单的音频转录任务。

**优势**：
- ✅ **支持异步调用**：Ollama 支持异步调用，非常适合高效处理大型数据集
- ✅ **本地处理**：无需 API 成本或速率限制
- ✅ **丰富的模型选择**：访问 [ollama.com](https://ollama.com/) 了解可用模型
- ✅ **隐私保护**：所有处理都在本地完成

**示例用法**：
```bash
# 使用 Ollama 处理图像
python main.py ./images ./output --type image --ollama-vision-model llava-llama3:latest --ollama-text-model llama3.2 --silent

# 使用 Ollama 进行 AU 提取
python main.py video.mp4 output/ --type AU --ollama-text-model llama3.2 --silent
```

#### 何时使用 ChatGPT/Gemini
**推荐用途**：高级视频分析、复杂的多模态推理以及高质量内容生成。

**优势**：
- ✅ **最先进的性能**：最新的 GPT-4o 和 Gemini 模型提供卓越的推理能力
- ✅ **高级视频理解**：对复杂视频分析和时间推理支持更好
- ✅ **高质量输出**：更细致、更详细的情感识别和推理
- ✅ **强大的多模态集成**：在文本、图像和视频模态上表现优异

**示例用法**：
```bash
python main.py video.mp4 output/ --type MER --chatgpt-model gpt-4o --silent

python main.py video.mp4 output/ --type MER --silent
```

**权衡**：存在 API 成本和速率限制，但通常为复杂的情感推理任务提供最高质量的结果。

#### 何时使用 Hugging Face 模型
**推荐用途**：当您需要最新的最先进的模型或 Ollama 不支持的特定功能时。

**自定义模型集成**：
如果您想使用最新的 Hugging Face 模型或 Ollama 不支持的功能：

1. **选项 1 - 自行实现**：导航至 `mer_factory/models/hf_models/__init__.py`，按照现有模式注册您的模型并实现所需功能。

2. **选项 2 - 请求支持**：在我们的仓库中提交问题，告诉我们您希望我们支持的模型，我们会考虑添加。

**当前支持的模型**：`google/gemma-3n-E4B-it` 以及其他在 HF 模型目录中列出的模型。

## 引用

如果您觉得 MER-Factory 对您的研究或项目有帮助，请考虑给我们一个 ⭐！您的支持将帮助我们成长并不断改进。

此外，如果您在工作中使用了 MER-Factory，请使用以下 BibTeX 条目引用我们：

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
