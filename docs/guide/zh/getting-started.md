---
layout: default
title: 快速开始
description: MER-Factory 安装和基本使用的快速入门指南
lang: zh
---

# MER-Factory 入门指南

只需几分钟，即可启动并运行 MER-Factory。本指南将引导您完成安装过程和您的第一个情绪识别管道。

## 系统概要

<div style="text-align: center;">
  <img src="../assets/framework.svg" style="border: none; width: 100%; max-width: 1000px;">
</div>

## 先决条件

在安装 MER-Factory 之前，请确保您的系统上已安装以下依赖项：

### 1. FFmpeg 安装

视频和音频处理需要 FFmpeg。

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
    <p>从 <a href="https://ffmpeg.org/download.html">ffmpeg.org</a> 下载</p>
  </div>
</div>

**验证安装：**
```bash
ffmpeg -version
ffprobe -version
```

### 2. OpenFace 安装
面部动作单元提取需要 [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki)。

#### Mac/Linux 安装方式

[参考代码](https://github.com/Lum1104/MER-Factory/blob/main/examples/MER_Factory.ipynb)

<div class="alert alert-info">
<strong>注意：</strong> 构建 OpenFace 后，请记下 <code>FeatureExtraction</code> 可执行文件的路径（通常在 <code>build/bin/FeatureExtraction</code>）。您将在配置时需要它。
</div>

#### Windows 安装与配置指南
1. 下载适用于 Windows 的[预编译版本](https://github.com/TadasBaltrusaitis/OpenFace/releases)。
2. 在 Powershell 中运行`download_models.ps1`，如无法访问 Dropbox 则按照 [模型下载文档](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download)下载模型放置到 `model/patch_experts` 路径下
3. 将 `FeatureExtraction.exe` 文件的**绝对路径**填入 `.env`


## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/Lum1104/MER-Factory.git
cd MER-Factory
```

### 2. 设置 Python 环境

```bash
# 创建一个新的 conda 环境
conda create -n mer-factory python=3.12
conda activate mer-factory

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境

```bash
# 复制环境文件示例
cp .env.example .env
```

编辑 `.env` 文件并填入您的设置：

```env
# API 密钥 (可选 - 根据您偏好的模型选择)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# OpenFace 配置 (AU 和 MER 管道必需)
OPENFACE_EXECUTABLE=/absolute/path/to/OpenFace/build/bin/FeatureExtraction

# 可选: Ollama 本地模型配置
# OLLAMA_HOST=http://localhost:11434
```

<div class="alert alert-warning">
<strong>重要提示：</strong> <code>OPENFACE_EXECUTABLE</code> 路径必须是绝对路径，并指向实际的可执行文件。
</div>

## 您的第一个管道

让我们运行您的第一个情绪识别管道！

### 1. 准备您的媒体文件

创建一个包含视频文件的测试目录：

```bash
mkdir test_input
# 将您的视频文件复制到 test_input/your_video.mp4
```

### 2. 运行 MER 管道

```bash
# 使用默认 Gemini 模型的基本 MER 管道
python main.py test_input/ output/ --type MER --silent

# 调整阈值
python main.py test_input/ output/ --type MER --threshold 0.8 --silent
```

### 3. 检查结果

```bash
# 查看生成的文件
ls output/
# your_video_merr_data.json - 包含完整的分析
# your_video_au_data.csv - 面部动作单元数据
# your_video.wav - 提取的音频
# your_video_peak_frame.jpg - 关键情绪时刻的帧
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

## 模型选项

MER-Factory 支持多种 AI 模型。根据您的需求进行选择：

### Google Gemini (默认)
```bash
python main.py input/ output/ --type MER
```
- **最适用于：** 高质量的多模态分析
- **需要：** 在 `.env` 文件中配置 `GOOGLE_API_KEY`

### OpenAI ChatGPT
```bash
python main.py input/ output/ --type MER --chatgpt-model gpt-4o
```
- **最适用于：** 先进的推理和视频分析
- **需要：** 在 `.env` 文件中配置 `OPENAI_API_KEY`

### Ollama (本地模型)
```bash
# 首先，拉取模型
ollama pull llava-llama3:latest
ollama pull llama3.2

# 使用 Ollama 运行
python main.py input/ output/ --type MER \
  --ollama-vision-model llava-llama3:latest \
  --ollama-text-model llama3.2
```
- **最适用于：** 隐私保护、无 API 费用、异步处理
- **需要：** 本地安装 Ollama

### Hugging Face 模型
```bash
python main.py input/ output/ --type MER --huggingface-model google/gemma-3n-E4B-it
```
- **最适用于：** 最新的研究模型、自定义实现
- **注意：** 自动单线程处理

## 管道类型

### 快速管道比较

| 管道 | 输入 | 输出 | 用例 |
|----------|-------|---------|----------|
| **MER** | 视频/图像 | 完整的情绪分析 | 完整的多模态数据集 |
| **AU** | 视频 | 面部动作单元 | 面部表情研究 |
| **Audio** | 视频 | 语音 + 音调分析 | 音频情绪识别 |
| **Video** | 视频 | 视觉描述 | 视频理解 |
| **Image** | 图像 | 图像情绪分析 | 静态情绪识别 |

### 命令示例

```bash
# 仅提取动作单元
python main.py video.mp4 output/ --type AU

# 仅进行音频分析
python main.py video.mp4 output/ --type audio

# 仅进行视频描述
python main.py video.mp4 output/ --type video

# 图像分析 (对于图像输入会自动检测)
python main.py ./images/ output/ --type image

# 使用自定义设置的完整 MER
python main.py videos/ output/ \
  --type MER \
  --threshold 0.9 \
  --peak-dis 20 \
  --concurrency 8 \
  --silent
```

## 测试您的安装

运行内置测试以验证一切正常：

```bash
# 测试 FFmpeg 集成
python test/test_ffmpeg.py your_video.mp4 test_output/

# 测试 OpenFace 集成
python test/test_openface.py your_video.mp4 test_output/
```

## 常见问题与解决方案

### 找不到 FFmpeg
**症状：** 与 `ffmpeg` 相关的 `FileNotFoundError`

**解决方案：**
1. 验证 FFmpeg 是否已安装：`ffmpeg -version`
2. 检查它是否在您的系统 PATH 中
3. 在 Windows 上，将 FFmpeg 添加到系统 PATH

### 找不到 OpenFace 可执行文件
**症状：** 找不到 FeatureExtraction 可执行文件

**解决方案：**
1. 验证 `.env` 中的路径是否为绝对路径
2. 检查文件权限：`chmod +x FeatureExtraction`
3. 手动测试：`/path/to/FeatureExtraction -help`

### API 密钥错误
**症状：** `401 Unauthorized` 错误

**解决方案：**
1. 验证 `.env` 中的 API 密钥是否正确
2. 检查是否有额外的空格或字符
3. 确保您的 API 账户已启用计费

### 内存问题
**症状：** 处理大文件时出现内存不足错误

**解决方案：**
1. 减少并发数：`--concurrency 1`
2. 使用较小的视频文件进行测试
3. 关闭其他占用大量内存的应用程序

## 后续步骤

现在您已经成功运行 MER-Factory，可以探索这些高级功能：

- **[API 参考](/MER-Factory/zh/api-reference)** - 详细的函数文档
- **[示例](/MER-Factory/zh/examples)** - 真实世界的使用示例
- **[技术文档](/MER-Factory/zh/technical-docs)** - 系统架构详情

## 需要帮助？

- 🐛 在 [GitHub Issues](https://github.com/Lum1104/MER-Factory/issues) 上报告问题
- 💬 在 [GitHub Discussions](https://github.com/Lum1104/MER-Factory/discussions) 上参与讨论
- 📖 阅读 [技术文档](/MER-Factory/zh/technical-docs) 以深入了解

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