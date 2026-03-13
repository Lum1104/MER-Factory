---
layout: default
title: 技术文档
description: 深入了解 MER-Factory 的架构、设计模式和可扩展性
lang: zh
---

# MER-Factory: 技术文档

## 1. 引言与系统概述

MER-Factory 是一个基于 Python 的开源框架，专为情感计算社区设计，旨在为训练多模态大型语言模型（MLLMs）创建统一的数据集。该工具最初设计用于处理完整的的多模态视频，现已发展到同样支持单模态分析（例如，图像或音频标注）。它能自动提取多模态特征（面部、音频、视觉），并利用 LLM 生成详细的分析和情感推理摘要。

该框架支持两种主要的标注工作流程：
1.  **无监督（无标签）：** 工具分析媒体并从头开始生成描述和情感数据。
2.  **有监督（提供标签）：** 通过 `--label-file` 参数提供基准标签，工具可以执行更有针对性的分析，专注于为特定情感的发生生成理据或解释。

该系统在构建时充分考虑了模块化和可扩展性，利用状态驱动的图架构 (`langgraph`) 来管理复杂的处理工作流。这使得从简单的单模态分析到完整的端到端多模态合成，都能实现灵活的流水线执行，为数据集构建提供了一个易于使用和开放的框架。

### 核心技术
-   **CLI 框架:** 使用 `Typer` 实现简洁、强大的命令行界面。
-   **工作流管理:** 使用 `LangGraph` 将处理流水线定义和执行为一个有状态的图。
-   **面部分析:** 使用 `OpenFace` 提取面部动作单元（AUs）。
-   **媒体处理:** 使用 `FFmpeg` 进行音频/视频操作（提取、帧抓取）。
-   **AI/LLM 集成:** 采用可插拔架构，支持 `OpenAI (ChatGPT)`、`Google (Gemini)`、`Ollama`（本地模型）和 `Hugging Face` 模型。
-   **数据处理:** 使用 `pandas` 处理表格化的 AU 数据。
-   **并发处理:** 使用 `asyncio` 高效、并行地处理多个文件。

## 系统概要

<div style="text-align: center;">
  <img src="../assets/framework.svg" style="border: none; width: 100%; max-width: 1000px;">
</div>

---

## 2. 系统架构与执行流程

应用程序的执行由作为入口点的 `main.py` 进行编排。核心逻辑被构建为由 `main_orchestrator` 函数管理的两个主要阶段。

1.  **阶段一：特征提取：** 这个初步阶段会为所有输入文件并发运行。它使用外部工具（`FFmpeg`, `OpenFace`）来提取主要分析所需的原始特征。这包括将音频流提取为 `.wav` 文件，以及运行 OpenFace 生成包含逐帧 AU 数据的 `.csv` 文件。此阶段由 `utils.processing_manager.run_feature_extraction` 处理。

2.  **阶段二：主要处理：** 此阶段使用 `langgraph` 计算图执行主要的分析流水线。该图会单独处理每个文件，通过一系列执行特定任务的节点传递一个状态对象。此阶段由 `utils.processing_manager.run_main_processing` 处理。

### 2.1. 状态管理 (`state.py`)

整个处理流水线是有状态的。`MERRState` 类（一个 `total=False` 的 `TypedDict`）作为在图节点之间传递的中央数据载体。`total=False` 的使用至关重要，因为它允许状态在仅部分填充时仍然有效，这对于在不同流水线分支中流动的增量构建状态至关重要。

每个节点都可以从这个状态字典中读取和写入数据。这种设计将节点彼此解耦，因为它们只需要了解状态对象，而无需关心其前后节点。

**`MERRState` 中的关键字段分组如下：**
-   **核心配置:** `processing_type`, `models`, `verbose`, `cache`, `ground_truth_label`。
-   **路径管理:** `video_path`, `video_id`, `output_dir`, `audio_path`, `au_data_path`, `peak_frame_path`。
-   **面部分析参数与结果:** `threshold`, `peak_distance_frames`, `detected_emotions`, `peak_frame_info`。
-   **中间 LLM 描述:** `audio_analysis_results`, `video_description`, `image_visual_description`, `peak_frame_au_description`, `llm_au_description`。
-   **最终输出:** `final_summary`, `error`。

### 2.2. 计算图 (`graph.py`)

应用程序的核心是在 `create_graph` 中定义的 `StateGraph`。这个图定义了不同处理类型的可能执行路径。

-   **节点:** 每个节点都是一个执行特定原子操作（例如, `generate_audio_description`）的 Python 函数。该框架同时支持 `async` 和 `sync` 节点。这种双重支持至关重要，因为虽然基于 API 的模型（Gemini, OpenAI, Ollama）能从 `asyncio` 的高并发性中受益，但某些库（特别是用于本地 Hugging Face 模型的库）是同步操作的。该图会根据所选模型动态加载正确的节点类型。
-   **边:** 边连接节点，定义控制流程。
-   **条件边:** 该图使用路由函数（`route_by_processing_type` 等）根据当前的 `MERRState` 动态决定下一步。这种机制使得工具能够在同一个图结构内运行不同的流水线（例如 'AU' vs. 'MER'）。

### 2.3. 配置、缓存与错误处理

-   **配置 (`utils/config.py`):** `AppConfig` 数据类集中了所有源自 CLI 参数和 `.env` 变量的配置。它作为传递给编排器的单一事实来源，在处理开始前验证输入（例如，检查所选模型是否有效）。
-   **错误处理:** 图节点内的错误不会使整个应用程序崩溃。相反，节点会捕获自身的异常，并将错误消息放入 `MERRState` 的 `error` 字段中。随后的路由函数会检查此字段是否存在，并将流程导向一个终端的 `handle_error` 节点，从而允许应用程序优雅地终止对单个文件的处理并继续处理下一个文件。

---

## 3. 核心模块与功能

### 3.1. 命令行界面 (`main.py`)

使用 `Typer` 定义的 CLI 负责填充 `AppConfig` 对象。这实现了用户界面层与业务逻辑之间的清晰分离。它包括强大的类型检查、帮助信息以及对用户提供参数的验证。

### 3.2. 面部与情感分析 (`facial_analyzer.py`, `emotion_analyzer.py`)

这是情感识别能力的科学核心，基于面部动作编码系统（FACS），并受到 NeurIPS 2024 关于 Emotion-LLaMA 的论文方法的启发 [1]。

-   **OpenFace AU 类型:** 该逻辑关键地区分了 OpenFace 输出的两种 AU 类型：
    -   `_c` (分类): 一个二进制值（0 或 1），表示动作单元的*存在*与否。
    -   `_r` (回归): 一个连续值（通常为 0-5），表示存在的动作单元的*强度*。
-   **`emotion_analyzer.py`:**
    -   `EMOTION_TO_AU_MAP`: 将离散情绪映射到所需的 `_c` AU 组合。要将一种情绪纳入考虑，必须满足这些存在性 AU 的最低阈值。
    -   `analyze_emotions_at_peak`: 如果满足存在性标准，此函数会计算相应 `_r` AU 的平均强度，以对情绪的强度进行评分。
-   **`facial_analyzer.py`:**
    -   `FacialAnalyzer` 类接收来自 OpenFace 的 `.csv` 文件，并通过对所有 `_r` AU 值求和来计算每帧的 `overall_intensity` 分数。
    -   `get_chronological_emotion_summary`: 在 `overall_intensity` 时间序列上使用 `scipy.signal.find_peaks` 来识别显著的情感时刻。峰值之间的距离由 `--peak-dis` CLI 参数控制。这一更新策略基于 Emotion-LLaMA 方法，承认一个情感高峰帧可能在时间上与影响相应动作单元的特定口语词汇相关联。
    -   `get_overall_peak_frame_info`: 找到具有最高 `overall_intensity` 的单帧，作为 MER 流水线中详细多模态分析的焦点。

### 3.3. LLM 集成

#### 3.3.1. 模型抽象 (`mer_factory/models/__init__.py`)

`LLMModels` 类使用工厂模式为与不同 LLM 提供商的交互提供统一接口。它会检查 CLI 参数并初始化相应的客户端（`GeminiModel`, `ChatGptModel` 等）。这种抽象是框架可扩展性的关键，因为支持新的 LLM 提供商只需要添加一个新的模型类并更新工厂逻辑。

#### 3.3.2. 提示工程 (`prompts.py`)

`PromptTemplates` 类集中管理所有提示。在描述性提示（`describe_image`, `analyze_audio`）中使用的一个关键策略是明确指示：`DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION`（除原始文本描述外，不要提供任何其他回应）。这最大限度地减少了后处理的需求，并确保 LLM 输出是干净、结构化的数据，适合包含在数据集中。综合性提示则高度结构化，引导 LLM 扮演专家心理学家的角色，并将其分析分为不同的逻辑部分。

### 3.4. 数据导出 (`export.py`)

任何流水线运行的最终输出都是一个详细的 JSON 文件，具体内容取决于分析类型（`MER`、`AU`、`image` 等）。以 `MER` 流水线为例，该 JSON 文件结构丰富，包含以下内容：

* `source_path`: 源媒体文件的完整路径。
* `chronological_emotion_peaks`: 按时间顺序排列的情感高峰列表。
* `coarse_descriptions_at_peak`: 包含所有中间单模态描述（音频、视频、视觉、AU）的字典。
* `final_summary`: 来自 LLM 的最终综合推理结果。

`export.py` 脚本是一个多功能工具，用于处理这些输出数据。其主要功能是解析这些结构化 JSON 文件的目录，提取最相关的字段并将结果整合到一个单一的、适合分析的 `.csv` 文件中。

此外，该脚本还包括为大型语言模型（LLM）微调准备数据的功能。它可以将处理后的数据（无论是来自初始 JSON 文件还是现有的 CSV 文件）转换为结构化的 ShareGPT `JSON` 或 `JSONL` 格式，可用于训练框架（例如，LLaMA-Factory）。

---

## 4. 模型选择与高级工作流

### 4.1. 模型推荐

模型的选择取决于具体任务、预算和期望的质量。

-   **Ollama (本地模型):**
    -   **推荐用于:** 图像分析、AU 分析、文本处理和简单的音频转录。
    -   **优点:** 无 API 成本、保护隐私，并能高效地对大型数据集进行异步处理。非常适合不需要复杂时序推理的任务。

-   **ChatGPT/Gemini (基于 API 的 SOTA 模型):**
    -   **推荐用于:** 高级视频分析、复杂的多模态推理，以及生成最高质量的综合摘要。
    -   **优点:** 卓越的推理能力，尤其是在理解视频中的时序上下文方面。它们为完整的 `MER` 流水线提供最细致入微和详细的输出。
    -   **权衡:** 会产生 API 费用，并受速率限制的影响。

-   **Hugging Face (本地模型):**
    -   **推荐用于:** 希望试验最新开源模型或需要 Ollama 尚不具备的特定功能的用户。
    -   **注意:** 这些模型目前同步运行，因此并发数限制为 1。

### 4.2. 高级缓存工作流：“最佳组合”分析

`--cache` 标志启用了一个强大的工作流，允许您为每种模态使用最佳模型，然后将结果合并。由于单一模型可能有其局限性（例如，Ollama 不支持视频），您可以为每种模态使用最强的可用模型分别运行流水线，然后运行最终的 `MER` 流水线来综合结果。

**工作流示例:**
1.  **使用 Qwen2-Audio 运行音频分析：** 使用强大的 API 模型以获得最佳的转录和音调分析。
    这会在输出子目录中生成 `{sample_id}_audio_analysis.json`。

2.  **使用 Gemma 运行视频分析：** 使用另一个 SOTA 模型进行视频描述。
    这会生成 `{sample_id}_video_analysis.json`。

3.  **运行最终的 MER 综合：** 运行完整的 `MER` 流水线。`--cache` 标志将检测到前几个步骤中已存在的 JSON 文件，并跳过这些模态的分析，直接加载结果。它将只运行最后的 `synthesize_summary` 步骤，合并高质量、预先计算好的分析。
    ```bash
    python main.py video.mp4 output/ --type MER --chatgpt-model gpt-4o --cache
    ```
这种方法允许您使用针对每项特定任务最强大的模型来构建数据集，而不会在整个工作流中被锁定在单一提供商上。

此外，工具会在输出文件夹中创建一个隐藏的 `.llm_cache` 目录。该目录存储了每次 API 调用的详细信息，包括模型名称、发送的确切提示以及模型的响应。如果后续运行检测到相同的请求，它将直接从缓存中检索内容。这避免了重复的 API 调用，显著节省时间并降低成本。💰

---

## 5. 处理流水线 (图内流程)

以下描述了在 `langgraph` 图中为每种主要处理类型执行的节点序列。

### 5.1. `AU` 流水线
1.  `setup_paths`: 设置文件路径。
2.  `run_au_extraction`: (图外) 运行 OpenFace。
3.  `filter_by_emotion`: 使用 `FacialAnalyzer` 查找按时间顺序排列的情感高峰。结果存储在状态中。
4.  `save_au_results`: 将 `detected_emotions` 列表保存到 JSON 文件中。

### 5.2. `image` 流水线
1.  `setup_paths`: 设置路径。
2.  `run_image_analysis`:
    -   在单张图片上运行 OpenFace 以获取 AU 数据。
    -   使用 `FacialAnalyzer` 获得 `au_text_description`。
    -   调用 LLM 解释 AU (`llm_au_description`)。
    -   调用具有视觉能力的 LLM 以获取图像的视觉描述。
3.  `synthesize_image_summary`: 使用综合提示调用 LLM，结合 AU 分析和视觉描述来创建 `final_summary`。
4.  `save_image_results`: 将所有生成的描述和摘要保存到 JSON 文件中。

### 5.3. `MER` (完整) 流水线
这是最全面的流水线。
1.  `setup_paths`
2.  `extract_full_features`: (图外) FFmpeg 提取音频，OpenFace 提取 AUs。
3.  `filter_by_emotion`: 与 AU 流水线相同，找到所有情感高峰。
4.  `find_peak_frame`:
    -   使用 `FacialAnalyzer` 识别强度最高的单个高峰帧。
    -   将此帧提取为 `.png` 图像并将其路径保存到状态中。
5.  `generate_audio_description`:
    -   将提取的音频文件发送给 LLM 进行转录和音调分析。
6.  `generate_video_description`:
    -   将整个视频发送给具有视觉能力的 LLM 以获得一般内容摘要。
7.  `generate_peak_frame_visual_description`:
    -   将保存的高峰帧图像发送给视觉 LLM 以获得详细的视觉描述。
8.  `generate_peak_frame_au_description`:
    -   将来自高峰帧的 AUs 连同 `describe_facial_expression` 提示一起发送给 LLM。
9.  `synthesize_summary`:
    -   将所有生成的描述编译成一个上下文块，并与最终的综合提示一起发送给 LLM。
10. `save_mer_results`: 将最终的、全面的 JSON 对象写入磁盘。

---

## 6. 可扩展性

该框架被设计为可扩展的。

-   **添加新的 LLM:**
    1.  在 `mer_factory/models/` 中创建一个新的模型类，并实现所需的方法：
        - `describe_facial_expression`
        - `describe_image`
        - `analyze_audio`
        - `describe_video`
        - `synthesize_summary`
       
    2.  在 `mer_factory/models/__init__.py` 的工厂中添加逻辑，以便根据新的 CLI 参数实例化您的新类。

-   **添加新的流水线:**
    1.  在 `utils/config.py` 中定义一个新的 `ProcessingType` 枚举。
    2.  在 `mer_factory/nodes/` 中为您的流水线添加一个新的入口节点。
    3.  更新 `graph.py` 中的 `route_by_processing_type` 函数，以路由到您的新节点。
    4.  在 `graph.py` 的 `StateGraph` 中添加必要的节点和边，以定义您流水线的工作流程，并在可能的情况下将其连接到现有节点（例如，重用 `save_..._results` 节点）。

---

## 7. 已知局限性

承认该框架所利用的底层 MLLM 技术的局限性非常重要。近期研究的一个关键见解是，虽然模型在识别简单、主要情绪的触发因素方面正变得越来越熟练，但它们往往难以解释更复杂、多方面情绪状态背后的“为什么” [2]。它们对细微情感背景的推理能力仍是一个活跃的研究领域，而 MER-Factory 生成的数据集正是为了帮助社区应对这一挑战。

[1] Cheng, Zebang, et al. "Emotion-llama: Multimodal emotion recognition and reasoning with instruction tuning." Advances in Neural Information Processing Systems 37 (2024): 110805-110853.  
[2] Lin, Yuxiang, et al. "Why We Feel: Breaking Boundaries in Emotional Reasoning with Multimodal Large Language Models." arXiv preprint arXiv:2504.07521 (2025).