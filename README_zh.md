# 👉🏻 MER-Factory 👈🏻

<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>
<br>

<p align="center">
  <a href="https://lum1104.github.io/MER-Factory/" target="_blank">📖 项目文档</a>
</p>

<p align="center"> <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Reasoning-red"> <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Recognition-red"> <img src="https://zenodo.org/badge/1007639998.svg" alt="DOI"> </p>

<p align="center">
  <a href="https://lum1104.github.io/MER-Factory/zh/">
    <img src="docs/assets/logo.svg" width="700">
  </a>
 </p>

> [!IMPORTANT]
> ✍️ 挑战：多模态情感计算并非一步到位——而是一条支离破碎的流水线。从原始文件到训练模型，过程中充斥着繁琐的数据预处理、缓慢且不一致的人工标注，以及复杂的训练环境搭建。
> 
> 🏭 MER-Factory：将这整条工作流统一为一座“工厂”。我们自动化完成预处理与标注，生成高质量、带推理的样本数据，并直接衔接到模型训练。
> 
> 🚀 别再来回切换不同工具：让我们的工厂替你完成重活，你只需专注于研究。

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
  - [Hugging Face 客户端-服务端架构](#hugging-face-客户端-服务端架构)
  - [命令行选项](#命令行选项)
  - [处理类型](#处理类型)
  - [导出数据集](#导出数据集)
  - [评估结果](#评估结果)
- [模型支持](#模型支持)
  - [模型推荐](#模型推荐)
- [训练](#训练)
- [引用](#引用)

## Pipeline 结构

<details>
<summary>点击展开/折叠</summary>

暂时移除，可调用 `graph.py` 中的 `print(app.get_graph().draw_mermaid())` 查看

</details>

## 特性

-   **动作单元（AU）流程**：提取面部动作单元（AUs），并转化为描述性自然语言。
-   **音频分析流程**：提取音频、转录语音，并进行详细语音语调分析。
-   **视频分析流程**：生成视频内容与上下文的全面描述。
-   **图像分析流程**：为静态图像提供端到端情感识别，包含视觉描述与情感综合。
-   **完整 MER 流程**：端到端多模态流程，定位情感峰值时刻，分析所有模态（视觉、音频、面部），并综合生成整体情感推理总结。

查看示例输出：
-   [llava-llama3_llama3.2_merr_data.json](examples/llava-llama3_llama3.2_merr_data.json)
-   [gemini_merr.json](examples/gemini_merr.json)

## 安装

完整的 Linux 环境安装示例（包括 Google Colab）可在以下位置找到：
- [`examples/MER_Factory.ipynb`](examples/MER_Factory.ipynb)

<p align="center">
  📚 请访问 <a href="https://lum1104.github.io/MER-Factory/zh/" target="_blank">项目文档</a> 查看详细安装与使用说明。
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

### Hugging Face 客户端-服务端架构

当使用 `--huggingface-model` 指定 HF 模型时，MER-Factory 会通过一个轻量客户端转发调用至本地/远端的 API Server（实际承载 HF 模型）。这样可保持主环境整洁并便于扩缩容。

1) 启动 HF API Server（单独终端中）：

```bash
# 示例：在 7860 端口提供 Whisper base
python -m mer_factory.models.hf_api_server --model_id openai/whisper-base --host 0.0.0.0 --port 7860
```

2) 像往常一样运行 MER-Factory，并通过 ID 选择 HF 模型：

```bash
python main.py path_to_video/ output/ --type MER --huggingface-model openai/whisper-base --silent
```

### 数据整理与超参数调优仪表板

我们提供了交互式网页仪表板，便于数据整理与超参数调优。可测试不同提示词、保存与运行配置、并对生成数据进行评分。

启动命令：

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

### 评估结果

MER-Factory 包含了一个全面的无参考评估工具包，用于在无需人工评分的情况下评估生成标注的质量。

#### 基础评估
```bash
# 评估输出目录中的所有样本
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv
```

#### 高级评估选项
```bash
# 运行详细输出模式，查看详细的失败原因
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv --verbose

# 跳过写入单个样本评估文件
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv --no-write-per-sample
```

#### 评估指标
评估工具包提供多种质量指标：

- **🖼️ CLIP 图像分数**：图像与描述之间的视觉匹配度
- **🔊 CLAP 音频分数**：使用 LAION-CLAP 的音频-文本对齐度
- **😊 AU F1 分数**：面部表情准确性对比 OpenFace AUs
- **🔗 NLI 一致性**：跨模态逻辑一致性
- **🎙️ ASR WER**：语音识别质量对比 Whisper 基线
- **📝 文本质量**：差异性、重复性和可读性指标
- **🎯 综合分数**：结合所有指标的整体质量（0-100）

#### 评估输出
- **单样本**：每个样本目录中的 `evaluation.json` 文件
- **数据集级别**：包含排名和统计信息的 `evaluation_summary.csv`
- **控制台**：美观的进度条和表现最佳样本表格

详细评估文档请参见 [`tools/evaluate/README.md`](tools/evaluate/README.md)。

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

## 训练

本训练指南将引导您完成从**数据分析/标注**到**启动模型训练**的完整端到端流程。该流程分为两个主要阶段：

1.  **阶段一：自动化数据准备**：使用 `train.sh` 脚本，将 MER-Factory 的分析产出，一键转换为训练框架所需的标准数据集格式，并自动完成注册。
2.  **阶段二：交互式启动训练**：启动 LLaMA-Factory 的图形化界面（Web UI），在其中加载准备好的数据集，并自由配置所有训练参数。


### 准备工作

在开始之前，请确保您已完成以下环境准备：

1.  **初始化子模块**
   
    本项目使用 Git Submodule 来集成 LLaMA-Factory，以确保训练环境的版本一致性和可复现性。
    
    请在克隆本仓库后，运行以下命令来初始化并下载子模块：
    ```bash
    git submodule update --init --recursive
    ```

2.  **安装依赖**
   
    本项目和 LLaMA-Factory 子模块拥有各自独立的依赖环境，需要分别安装：
    ```bash
    # 1. 安装 MER-Factory 的主依赖
    pip install -r requirements.txt

    # 2. 安装 LLaMA-Factory 子模块的依赖
    pip install -r LLaMA-Factory/requirements.txt
    ```

### 阶段一：自动化数据准备

在您使用 MER-Factory 的 `main.py` 完成对原始数据的分析之后，就可以使用 `train.sh` 脚本来准备数据集了。

此脚本的核心任务是**自动化所有繁琐的数据准备工作**。它会读取 MER-Factory 的分析结果，将其转换为 LLaMA-Factory 所需的 ShareGPT 格式，并自动在 LLaMA-Factory 中完成注册。

#### 使用示例

为了保证实验的可追溯性和一致性，我们推荐使用以下格式为您的数据集命名：

`原始数据集_分析模型_任务类型`

处理 **MER** 任务的数据，并按规范命名数据集：
```bash
# 假设使用了 llava 和 llama3.2 两个分析模型
bash train.sh --file_type "image" --dataset_name "mer2025_llava_llama3.2_MER"
```

处理 **audio** 任务的数据，并按规范命名数据集：
```bash
# 假设使用了 gemini api 模型
bash train.sh --file_type "audio" --dataset_name "mer2025_gemini_audio"
```

处理 **video** 任务的数据，并按规范命名数据集：
```bash
# 假设使用了 gemini api 模型
bash train.sh --file_type "video" --dataset_name "mer2025_gemini_video"
```

处理 **image** 任务的数据，并按规范命名数据集：
```bash
# 假设使用了 chatgpt gpt-4o 模型
bash train.sh --file_type "mer" --dataset_name "mer2025_gpt-4o_image"
```

脚本成功运行后，您的数据集（如：mer2025_llava_llama3.2_MER）就已经准备就绪，并注册到了 LLaMA-Factory 的 dataset_info 中，可以在下一阶段直接使用。

### 阶段二：启动训练 (启动 LLaMA-Factory Web UI)

当您的数据集准备好后，就可以启动 LLaMA-Factory 的图形化界面来配置和开始您的训练任务。

1. **进入 LLaMA-Factory 目录**
   
   ```bash
   cd LLaMA-Factory
   ```
2. **启动 Web UI**
   
   ```bash
   llamafactory-cli webui
   ```
3. **在 Web UI 中配置并训练**



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
