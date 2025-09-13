# 添加自定义Hugging Face模型指南

本指南将详细说明如何在MER-Factory项目中添加新的Hugging Face单模态模型支持。

## 📋 概述

MER-Factory支持多种类型的AI模型，包括：
- **多模态模型**：支持文本、图像、音频、视频（如Qwen2.5-Omni）
- **单模态模型**：专门处理特定类型的数据（如Qwen2-Audio）
- **文本模型**：纯文本生成和分析（如LLaMA、Mistral）

## 🛠️ 实现步骤

### 步骤1：创建模型类

1. **复制模板文件**：
   ```bash
   cp mer_factory/models/hf_models/custom_text_model.py mer_factory/models/hf_models/your_model.py
   ```

2. **修改类名和模型ID**：
   ```python
   class YourModelName:  # 修改类名
       def __init__(self, model_id: str, verbose: bool = True):
           # 修改为你的实际模型ID
           # 例如: "your-org/your-model-name"
   ```

3. **根据模型类型调整实现**：
   - **文本模型**：使用 `AutoModelForCausalLM` 和 `AutoTokenizer`
   - **图像模型**：使用相应的视觉模型类
   - **音频模型**：使用音频处理相关的模型类

### 步骤2：注册模型

在 `mer_factory/models/hf_models/__init__.py` 中添加你的模型：

```python
HUGGINGFACE_MODEL_REGISTRY = {
    # ... 现有模型 ...
    "your-org/your-model-name": (".your_model", "YourModelName"),
}
```

### 步骤3：实现必需方法

每个模型类必须实现以下标准接口：

```python
def describe_facial_expression(self, prompt: str) -> str:
    """分析面部表情"""
    pass

def describe_image(self, image_path: Path, prompt: str) -> str:
    """分析图像（如果不支持则返回提示信息）"""
    pass

def analyze_audio(self, audio_path: Path, prompt: str) -> str:
    """分析音频（如果不支持则返回提示信息）"""
    pass

def describe_video(self, video_path: Path, prompt: str) -> str:
    """分析视频（如果不支持则返回提示信息）"""
    pass

def synthesize_summary(self, prompt: str) -> str:
    """生成文本摘要"""
    pass
```

### 步骤4：测试集成

1. **运行测试脚本**：
   ```bash
   python test_custom_model.py
   ```

2. **测试实际使用**：
   ```bash
   python main.py your_input.mp4 output/ --huggingface-model your-org/your-model-name
   ```

## 📝 模型类型示例

### 文本模型示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomTextModel:
    def __init__(self, model_id: str, verbose: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def _generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 图像模型示例

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

class CustomImageModel:
    def __init__(self, model_id: str, verbose: bool = True):
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    def describe_image(self, image_path: Path, prompt: str) -> str:
        image = Image.open(image_path)
        inputs = self.processor(image, prompt, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=512)
        return self.processor.decode(out[0], skip_special_tokens=True)
```

## 🔧 高级配置

### 设备管理

```python
def _initialize_pipeline(self):
    # 自动检测最佳设备
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 根据设备调整数据类型
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
```

### 内存优化

```python
def _initialize_pipeline(self):
    # 使用8bit量化减少内存使用
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_id,
        load_in_8bit=True,  # 8bit量化
        device_map="auto"
    )
```

### 错误处理

```python
def _generate_text(self, prompt: str) -> str:
    try:
        # 生成逻辑
        return result
    except torch.cuda.OutOfMemoryError:
        console.log("[yellow]GPU内存不足，尝试使用CPU...[/yellow]")
        # 降级到CPU处理
        return self._generate_text_cpu(prompt)
    except Exception as e:
        console.log(f"[red]生成失败: {e}[/red]")
        return ""
```

## 🚀 使用示例

### 命令行使用

```bash
# 使用自定义文本模型进行情感分析
python main.py video.mp4 output/ --huggingface-model meta-llama/Llama-2-7b-chat-hf

# 使用自定义图像模型
python main.py image.jpg output/ --huggingface-model your-org/your-image-model
```

### 编程使用

```python
from mer_factory.models import LLMModels

# 初始化模型
models = LLMModels(
    huggingface_model_id="your-org/your-model-name",
    verbose=True
)

# 使用模型
result = models.model_instance.describe_facial_expression("分析这个面部表情")
print(result)
```

## 📚 支持的模型类型

### 文本生成模型
- LLaMA系列：`meta-llama/Llama-2-7b-chat-hf`
- Mistral系列：`mistralai/Mistral-7B-Instruct-v0.2`
- DialoGPT：`microsoft/DialoGPT-medium`
- GPT-2：`gpt2`

### 图像理解模型
- BLIP：`Salesforce/blip-image-captioning-base`
- CLIP：`openai/clip-vit-base-patch32`
- Vision Encoder-Decoder：`nlpconnect/vit-gpt2-image-captioning`

### 音频处理模型
- Whisper：`openai/whisper-base`
- Wav2Vec2：`facebook/wav2vec2-base`

## ⚠️ 注意事项

1. **内存要求**：大型模型需要足够的GPU内存
2. **依赖管理**：确保安装模型所需的所有依赖
3. **模型兼容性**：某些模型可能需要特定的transformers版本
4. **性能优化**：考虑使用量化或模型并行来提高效率

## 🆘 故障排除

### 常见问题

1. **CUDA内存不足**：
   - 使用较小的模型
   - 启用8bit或4bit量化
   - 使用CPU推理

2. **模型加载失败**：
   - 检查网络连接
   - 验证模型ID是否正确
   - 确认transformers版本兼容性

3. **生成质量差**：
   - 调整生成参数（temperature, top_p等）
   - 优化提示词模板
   - 尝试不同的模型变体

## 📞 获取帮助

如果遇到问题，可以：
1. 查看项目GitHub Issues
2. 参考transformers官方文档
3. 检查模型在Hugging Face Hub上的说明

---

**祝你在MER-Factory中成功集成新的Hugging Face模型！** 🎉
