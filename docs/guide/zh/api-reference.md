---
layout: default
title: API 参考
description: MER-Factory 类和函数的完整 API 参考文档
lang: zh
---

# API 参考

MER-Factory 核心类、函数和模块的完整参考文档。

## 核心模块

### `main.py` - CLI 入口点

MER-Factory 命令行界面的主入口点，使用 Typer 构建。

#### 函数

##### `main_orchestrator(config: AppConfig) -> None`

主编排函数，协调整个处理流程。它由 `process` 命令调用。

**参数:**

* `config` (`AppConfig`): 一个应用程序配置对象，包含从 CLI 参数派生的所有处理参数。

**处理流程:**

1.  验证模型选择和必需的路径。
2.  如果提供了标签文件，则加载地面实况标签。
3.  初始化所需的 LLM 模型。
4.  发现在输入路径中所有待处理的媒体文件。
5.  **阶段 1:** 并发运行特征提取阶段（音频使用 `FFmpeg`，面部动作单元使用 `OpenFace`）。
6.  **阶段 2:** 使用 `LangGraph` 图执行主处理流程。
7.  处理多个文件的并发处理，支持同步（Hugging Face）和异步（OpenAI, Ollama）模型。
8.  打印成功、跳过和失败文件的最终摘要。

##### `process(...)`

作为 CLI 面向用户的入口点的 Typer 命令。它收集所有用户输入并将其传递给 `main_orchestrator`。

**参数和选项:**

* `input_path` (Path): **\[必需]** 单个媒体文件或包含媒体文件的目录的路径。
* `output_dir` (Path): **\[必需]** 所有输出（CSV、日志等）将被保存的目录。
* `--type` / `-t` (ProcessingType): 要执行的处理类型。默认为 `MER` (多模态情绪识别)。
* `--label-file` / `-l` (Path): 包含 `name` 和 `label` 列的 CSV 文件的可选路径，用于地面实况数据。
* `--threshold` / `-th` (float): 从动作单元检测情绪的强度阈值。默认为 `0.8`。
* `--peak_dis` / `-pd` (int): 两个检测到的情绪峰值之间的最小距离（以帧为单位）。默认为 `15`。
* `--concurrency` / `-c` (int): 并发处理的文件数量。默认为 `4`。
* `--ollama-vision-model` / `-ovm` (str): 要使用的 Ollama 视觉模型的名称 (例如, `llava`)。
* `--ollama-text-model` / `-otm` (str): 要使用的 Ollama 文本模型的名称 (例如, `llama3`)。
* `--chatgpt-model` / `-cgm` (str): 要使用的 OpenAI 模型的名称 (例如, `gpt-4o`)。
* `--huggingface-model` / `-hfm` (str): 要使用的 Hugging Face 模型的模型 ID。
* `--silent` / `-s` (bool): 如果设置，则以最少的控制台输出运行 CLI。默认为 `False`。
* `--cache` / `-ca` (bool): 如果设置，则重用先前运行中已有的特征提取结果。默认为 `False`。

### `facial_analyzer.py` - 面部分析

此模块包含 `FacialAnalyzer` 类，负责处理来自 OpenFace 的输出。

#### 类: `FacialAnalyzer`

分析完整的 OpenFace 动作单元 (AU) 数据文件，以找到情绪峰值，识别整体最强烈的帧，并生成摘要。

##### `__init__(self, au_data_path: Path)`

通过加载和预处理 OpenFace CSV 文件来初始化分析器。

* **参数:**
    * `au_data_path` (Path): OpenFace 输出的 CSV 文件的路径。
* **引发:**
    * `FileNotFoundError`: 如果 CSV 文件不存在。
    * `ValueError`: 如果 CSV 为空或不包含 AU 强度列 (`_r` 后缀)。

##### `get_chronological_emotion_summary(self, peak_height=0.8, peak_distance=20, emotion_threshold=0.8)`

在视频的时间轴中找到所有显著的情绪峰值，并为每个峰值生成人类可读的摘要。

* **参数:**
    * `peak_height` (float): 注册为峰值所需的最小“整体强度”。默认为 `0.8`。
    * `peak_distance` (int): 峰值之间的最小帧数。默认为 `20`。
    * `emotion_threshold` (float): 在一个峰值处，认为存在某种情绪所需的最小 AU 强度分数。默认为 `0.8`。
* **返回:**
    * `tuple`: 一个元组，包含：
        * `list`: 一个字符串列表，其中每个字符串描述一个情绪峰值 (例如, `"Peak at 3.45s: happy (moderate), surprise (slight)"`)。如果未找到峰值，则返回 `["neutral"]`。
        * `bool`: 一个标志，指示视频是否被认为具有表现力（即，是否检测到任何有效的情绪峰值）。

##### `get_overall_peak_frame_info(self)`

根据 AU 强度的总和，在整个视频中找到情绪最强烈的单个帧。

* **返回:**
    * `dict`: 一个包含有关峰值帧信息的字典，包括 `frame_number`、`timestamp` 以及最活跃的 AU 及其强度的字典 (`top_aus_intensities`)。

##### `get_frame_au_summary(self, frame_index=0, threshold=0.8)`

获取特定帧的动作单元摘要。这主要用于单图像分析。

* **参数:**
    * `frame_index` (int): 要分析的帧的索引。默认为 `0`。
    * `threshold` (float): AU 被认为是活跃的最小强度。默认为 `0.8`。
* **返回:**
    * `str`: 对指定帧中活跃 AU 的人类可读描述。

### `emotion_analyzer.py` - 情绪和 AU 逻辑

该模块提供了根据既定映射将动作单元 (AUs) 解释为情绪的核心逻辑。

#### 类: `EmotionAnalyzer`

一个实用工具类，提供静态方法来集中处理从 OpenFace 数据分析情绪和动作单元的逻辑。

##### `analyze_emotions_at_peak(peak_frame, emotion_threshold=0.8)`

分析单个数据帧（通常是情绪峰值）中存在的情绪。

* **参数:**
    * `peak_frame` (pd.Series): Pandas DataFrame 中对应于峰值帧的一行。
    * `emotion_threshold` (float): 构成一种情绪的 AU 的最小平均强度分数，以认为该情绪存在。默认为 `0.8`。
* **返回:**
    * `list`: 一个字典列表。每个字典包含检测到的情绪的详细信息，包括 `emotion`、`score` 和 `strength`（轻微、中等或强烈）。

##### `get_active_aus(frame_data, threshold=0.8)`

从单个帧的数据中提取活跃动作单元及其强度的字典。

* **参数:**
    * `frame_data` (pd.Series): DataFrame 中的一行。
    * `threshold` (float): AU 被认为是活跃的最小强度。默认为 `0.8`。
* **返回:**
    * `dict`: 一个将活跃 AU 代码（例如，`AU06_r`）映射到其强度值的字典。

##### `extract_au_description(active_aus)`

从活跃 AU 的字典生成人类可读的字符串。

* **参数:**
    * `active_aus` (dict): 由 `get_active_aus` 返回的活跃 AU 代码及其强度的字典。
* **返回:**
    * `str`: 一个格式化的字符串，描述活跃的 AU 及其强度 (例如, "Cheek raiser (intensity: 2.50), Lip corner puller (smile) (intensity: 1.80)")。如果输入字典为空，则返回 "Neutral expression."。

#### 常量

* `AU_TO_TEXT_MAP`: 一个将 OpenFace 动作单元代码 (例如, `"AU01_r"`) 映射到人类可读描述 (例如, `"Inner brow raiser"`) 的字典。
* `EMOTION_TO_AU_MAP`: 一个将基本情绪 (例如, `"happy"`) 映射到构成该情绪的主要动作单元 (存在, `_c`) 列表的字典。这是用于情绪检测的核心映射。

---

## 实用工具模块

### `utils/config.py` - 应用程序配置

该模块使用 Pydantic 定义了应用程序的主要配置结构，确保所有设置都经过验证且类型正确。

#### 类: `AppConfig`

一个 Pydantic `BaseModel`，用于保存和验证从 CLI 传递的所有应用程序设置。

**属性:**

* `input_path` (Path): 输入文件或目录的路径。
* `output_dir` (DirectoryPath): 结果将被保存的目录路径。
* `processing_type` (ProcessingType): 一个枚举，指示要执行的分析类型 (`MER`, `AU`, `audio` 等)。
* `error_logs_dir` (Path): 用于存储错误日志的目录路径，在 `output_dir` 内自动创建。
* `label_file` (Optional\[FilePath]): 带有地面实况标签的 CSV 文件的可选路径。
* `threshold` (float): 情绪检测的强度阈值。
* `peak_distance_frames` (int): 检测到的情绪峰值之间的最小帧距。
* `silent` (bool): 抑制详细控制台输出的标志。
* `cache` (bool): 启用中间结果缓存的标志。
* `concurrency` (int): 并行处理的文件数。
* `ollama_vision_model` (Optional\[str]): 要使用的 Ollama 视觉模型的名称。
* `ollama_text_model` (Optional\[str]): 要使用的 Ollama 文本模型的名称。
* `chatgpt_model` (Optional\[str]): 要使用的 OpenAI 模型的名称。
* `huggingface_model_id` (Optional\[str]): 要使用的 Hugging Face 模型的 ID。
* `labels` (Dict\[str, str]): 用于存储加载的地面实况标签的字典。
* `openface_executable` (Optional\[str]): OpenFace 可执行文件的路径，从 `.env` 文件加载。
* `openai_api_key` (Optional\[str]): OpenAI API 密钥，从 `.env` 文件加载。
* `google_api_key` (Optional\[str]): Google API 密钥，从 `.env` 文件加载。

**方法:**

* `get_model_choice_error() -> Optional[str]`: 验证是否至少配置了一个 LLM。
* `get_openface_path_error() -> Optional[str]`: 在需要时检查 OpenFace 可执行文件路径是否有效。

#### 枚举: `ProcessingType`

一个字符串 `Enum`，定义了可以执行的有效处理类型。

**成员:**

* `AU`: 仅动作单元分析。
* `AUDIO`: 仅音频分析。
* `VIDEO`: 仅视频描述。
* `MER`: 完整的多模态情绪识别。
* `IMAGE`: 仅图像分析。

### `utils/file_handler.py` - 文件系统操作

该模块包含用于发现文件和从文件系统加载数据的辅助函数。

#### 函数

##### `find_files_to_process(input_path: Path, verbose: bool = True) -> List[Path]`

从给定的输入路径递归地查找所有有效的媒体文件（视频、图像、音频）。

* **参数:**
    * `input_path` (Path): 单个文件或目录的路径。
    * `verbose` (bool): 如果为 `True`，则打印详细输出。
* **返回:**
    * `List[Path]`: 所有待处理文件的 `Path` 对象列表。
* **引发:**
    * `SystemExit`: 如果输入路径无效或未找到可处理的文件。

##### `load_labels_from_file(label_file: Path, verbose: bool = True) -> Dict[str, str]`

从指定的 CSV 文件加载地面实况标签。CSV 必须包含 `name` 和 `label` 列。

* **参数:**
    * `label_file` (Path): CSV 文件的路径。
    * `verbose` (bool): 如果为 `True`，则打印详细输出。
* **返回:**
    * `Dict[str, str]`: 将文件词干（不带扩展名的名称）映射到其地面实况标签的字典。
* **引发:**
    * `SystemExit`: 如果文件未找到或无法正确解析。

### `utils/processing_manager.py` - 处理与特征提取

该模块管理主要处理任务的执行流程，包括特征提取和处理并发。

#### 函数

##### `build_initial_state(file_path: Path, config: AppConfig, models: LLMModels) -> MERRState`

为单个文件构建初始状态字典，该字典将传递给处理图。

* **参数:**
    * `file_path` (Path): 正在处理的媒体文件的路径。
    * `config` (AppConfig): 应用程序配置对象。
    * `models` (LLMModels): 初始化的 LLM 模型对象。
* **返回:**
    * `MERRState`: 一个包含图的初始状态的 TypedDict。

##### `run_main_processing(...) -> Dict[str, int]`

管理主处理循环。它检查缓存的最终输出，并根据模型的能力分派同步或异步运行的作业。

* **参数:**
    * `files_to_process` (List\[Path]): 要处理的文件列表。
    * `graph_app` (Any): 编译后的 `langgraph` 应用程序。
    * `initial_state_builder` (functools.partial): 用于构建初始状态的部分函数。
    * `config` (AppConfig): 应用程序配置对象。
    * `is_sync` (bool): 指示是否以同步模式运行的标志。
* **返回:**
    * `Dict[str, int]`: 一个总结结果（`success`、`failure`、`skipped`）的字典。

##### `run_feature_extraction(files_to_process: List[Path], config: AppConfig)`

在主处理开始前，对文件异步运行必要的特征提取工具（`FFmpeg` 用于音频，`OpenFace` 用于面部 AU）。

* **参数:**
    * `files_to_process` (List\[Path]): 需要特征提取的文件列表。
    * `config` (AppConfig): 应用程序配置对象。

---

## 代理与图模块

### `mer_factory/state.py` - 流程状态

该模块定义了在处理图中所有节点之间传递的共享状态对象。

#### 类: `MERRState`

一个 `TypedDict`，表示 MERR 流程的增量构建状态。使用 `total=False` 允许状态在仅部分填充时也有效，因为它在图中流动。状态分为以下几个部分：

* **核心配置与设置:** `processing_type`, `models`, `verbose`, `cache`, `ground_truth_label`。
* **文件与路径管理:** `video_path`, `video_id`, `output_dir`, `audio_path`, `au_data_path`, `peak_frame_path` 等。
* **面部与情绪分析结果:** `threshold`, `peak_distance_frames`, `detected_emotions`, `peak_frame_info`, `peak_frame_au_description`。
* **多模态描述结果:** `audio_analysis_results`, `video_description`, `image_visual_description`。
* **仅图像流程结果:** `au_text_description`, `llm_au_description`。
* **最终综合与错误处理:** `final_summary`, `error`。

### `mer_factory/graph.py` - 处理图

该模块使用 `langgraph.StateGraph` 构建处理流程。它定义了所有的节点、边和条件路由逻辑。

#### 函数

##### `create_graph(use_sync_nodes: bool = False)`

创建并编译模块化的 MERR 构建图。它根据 `use_sync_nodes` 标志动态导入异步或同步节点函数，该标志由所选的 LLM 决定（Hugging Face 模型同步运行）。

该函数将所有节点添加到图中，设置入口点，并使用边和条件路由器定义完整的控制流。

##### 路由函数

一系列小函数根据当前的 `MERRState` 决定图中的下一步。

* `route_by_processing_type`: 设置后的主路由器。它将流程引导到所选流程（`MER`, `AU`, `audio`, `video`, 或 `image`）的入口节点。
* `route_after_emotion_filter`: 在 AU 分析之后，此路由要么保存结果（对于 `AU` 流程），要么继续 `MER` 流程的下一步。
* `route_after_audio_generation`: 在音频分析之后，此路由要么保存结果（对于 `audio` 流程），要么继续 `MER` 流程的下一步。
* `route_after_video_generation`: 在视频分析之后，此路由要么保存结果（对于 `video` 流程），要么继续 `MER` 流程的下一步。

### `mer_factory/prompts.py` - 提示模板

该模块集中了所有用于与 LLM 交互的提示模板。

#### 类: `PromptTemplates`

一个将所有提示模板存储和管理为静态方法的类。

##### `describe_facial_expression()`

返回一个提示，指示 LLM 扮演面部动作编码系统 (FACS) 专家的角色，并根据一系列动作单元描述面部表情。

##### `describe_image(has_label: bool = False)`

返回一个用于分析图像的提示。
* 如果 `has_label` 为 `True`，提示将包含地面实况标签，并要求对主要主体、服装和场景进行客观描述。
* 如果为 `False`，它会要求进行相同的客观描述，但不带标签。

##### `analyze_audio(has_label: bool = False)`

返回一个用于分析音频文件的提示。它要求完成两项任务：语音转录和音频特征（音调、音高等）的描述。
* 如果 `has_label` 为 `True`，地面实况标签将包含在提示中。

##### `describe_video(has_label: bool = False)`

返回一个用于描述视频内容的提示，侧重于客观的视觉元素、人物和动作。
* 如果 `has_label` 为 `True`，地面实况标签将包含在提示中。

##### `synthesize_summary(has_label: bool = False)`

返回一个用于最终分析的提示，该提示综合了所有收集到的多模态线索。
* 如果 `has_label` 为 `True`，该提示会要求 LLM 扮演心理学家的角色，并根据证据建立一个基本原理来解释为什么基准标签是正确的。
* 如果 `has_label` 为 `False`，该提示会要求 LLM 推断主体的请感状态，并根据线索叙述其情绪历程。

### `mer_factory/models/__init__.py` - LLM 工厂

该模块提供了一个工厂类，用于根据提供的 CLI 参数初始化正确的 LLM。

#### 类: `LLMModels`

一个使用字典分派模式来选择和初始化适当的 LLM（例如, ChatGPT, Ollama, Hugging Face, Gemini）的类。

##### `__init__(self, api_key: str, ollama_text_model_name: str, ollama_vision_model_name: str, chatgpt_model_name: str, huggingface_model_id: str, verbose: bool)`

根据提供的参数初始化模型实例。它会遍历一个支持的模型类型列表，并实例化第一个满足必要条件的模型。

* **参数:**
    * `api_key` (字符串, 可选): 用于 OpenAI 或 Gemini 等服务的通用 API 密钥。
    * `ollama_text_model_name` (字符串, 可选): Ollama 文本模型的名称。
    * `ollama_vision_model_name` (字符串, 可选): Ollama 视觉模型的名称。
    * `chatgpt_model_name` (字符串, 可选): ChatGPT 模型的名称 (例如, 'gpt-4o')。
    * `huggingface_model_id` (字符串, 可选): Hugging Face 模型的 ID。
    * `verbose` (布尔值): 是否打印详细日志。默认为 `True`。
* **引发:**
    * `ValueError`: 如果因为没有提供所需的参数（例如，模型名称和/或 API 密钥）而无法初始化模型。

### `mer_factory/nodes/async_nodes.py` & `mer_factory/nodes/sync_nodes.py` - 异步与同步执行

本节概述了异步和同步节点实现之间的关键区别。它们之间的选择由所选的 LLM 决定，因为某些模型（如 Hugging Face）需要同步执行。

#### 关键区别 ⚖️

`sync_nodes.py` 和 `async_nodes.py` 之间的主要区别在于它们的执行模型。`sync_nodes.py` 包含为**同步（顺序）执行**设计的函数，这对于像 Hugging Face 这样可能不支持异步操作的模型是必需的。相比之下，`async_nodes.py` 是为**异步（并发）执行**而构建的，它通过允许程序在等待 I/O 操作完成时处理其他任务来提高性能。

#### 具体区别

* **函数定义**: `async_nodes.py` 中的所有函数都使用 `async def` 语法定义，并使用 `await` 进行非阻塞调用。`sync_nodes.py` 对其所有函数使用标准的 `def`。

* **处理阻塞操作**: `async_nodes.py` 使用 `asyncio.to_thread()` 在单独的线程中运行阻塞性 I/O 操作，例如保存文件或初始化 `FacialAnalyzer` 类。这可以防止这些操作阻塞整个事件循环。同步版本直接执行这些任务。

* **并发 API 调用**: 在 `async_nodes.py` 中，`run_image_analysis` 函数使用 `asyncio.gather` 并发执行两个独立的 API 调用，以获取 LLM 对动作单元（AUs）的描述和整体视觉描述。同步版本则顺序执行这些调用。

---

## API 与本地模型实现

该应用程序支持多种模型，这些模型可以分为基于 API（需要远程服务）或本地（在用户机器上运行）两类。

### Gemini Model (基于API)

`GeminiModel` 类处理与 Google Gemini API 的所有交互。

* **模型**: 它使用单一的多模态模型 `gemini-2.0-flash-lite` 来完成所有任务，包括文本、图像、音频和视频分析。同一个模型实例用于通用和视觉特定任务。
* **初始化**: 该模型使用 Google API 密钥进行初始化。
* **功能**:
    * 它可以通过将媒体文件编码为 base64 并随提示一起发送来分析文本、图像、音频和视频。
    * 与 API 的所有交互都是异步的。

### Ollama Model (混合本地/API)

`OllamaModel` 类提供了一种混合方法，使用本地 Ollama 模型进行文本和视觉任务，同时利用本地 Hugging Face 模型进行音频处理。

* **模型**:
    * 它可以为文本和视觉任务配置独立的 Ollama 模型。如果只提供了视觉模型，它也用于基于文本的任务。
    * 对于音频，它使用 Hugging Face 流水线，特别是 `"openai/whisper-base"`。
* **初始化**: 通过提供 Ollama 文本和视觉模型的名称进行初始化。Hugging Face 音频模型在检测到的设备（CUDA, MPS, 或 CPU）上进行初始化。
* **功能**:
    * 文本和图像生成调用是异步的。
    * 音频分析使用 Hugging Face 流水线同步执行。
    * 明确不支持视频分析，并将返回一个空字符串。

### Hugging Face Models (本地)

该应用程序通过动态加载系统支持各种特定的、本地运行的 Hugging Face 模型。

* **模型注册表**: `HUGGINGFACE_MODEL_REGISTRY` 包含一个支持模型的字典，将模型 ID 映射到其特定的模块和类名。
    * 示例包括 `"google/gemma-3n-E4B-it"`, `"Qwen/Qwen2-Audio-7B-Instruct"`, 和 `"zhifeixie/Audio-Reasoner"`。
* **动态加载**: `get_hf_model_class` 函数提供“懒加载”功能，仅在请求给定模型时才导入其必要的模块。这避免了依赖冲突和不必要的内存使用。
* **错误处理**: 如果请求了不支持的模型 ID，系统将引发 `ValueError`。

---