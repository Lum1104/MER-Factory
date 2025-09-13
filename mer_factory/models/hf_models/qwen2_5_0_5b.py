import torch
from pathlib import Path
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

console = Console(stderr=True)


class Qwen2_5_0_5BModel:
    """
    Qwen2.5-0.5B-Instruct模型的包装器，基于Hugging Face Transformers。
    这是一个支持中文的轻量级文本生成模型，适合中文情感分析任务。
    """

    def __init__(self, model_id: str, verbose: bool = True):
        """
        初始化Qwen2.5-0.5B-Instruct模型。

        Args:
            model_id (str): Hugging Face模型ID，默认使用 'Qwen/Qwen2.5-0.5B-Instruct'
            verbose (bool): 是否打印详细日志
        """
        self.model_id = model_id or "Qwen/Qwen2.5-0.5B-Instruct"
        self.verbose = verbose
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """加载Qwen2.5-0.5B-Instruct模型和分词器"""
        if self.verbose:
            console.log(f"正在初始化Qwen2.5-0.5B-Instruct模型 '{self.model_id}'...")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            
            # 设置pad_token（如果不存在）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # 如果使用CPU，手动移动模型
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            console.log(f"Qwen2.5-0.5B-Instruct模型 '{self.model_id}' 初始化成功，设备: {self.model.device}")
            
        except Exception as e:
            console.log(f"[bold red]错误: 无法初始化Qwen2.5-0.5B-Instruct模型: {e}[/bold red]")
            raise

    def _generate_text(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        内部方法：运行Qwen2.5-0.5B-Instruct文本生成管道
        
        Args:
            prompt (str): 输入提示文本
            max_new_tokens (int): 最大生成token数
            
        Returns:
            str: 生成的文本
        """
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048  # Qwen2.5支持更长的上下文
            )
            
            # 移动到正确的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成文本
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # 解码输出（只保留新生成的部分）
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            console.log(f"[bold red]❌ Qwen2.5-0.5B-Instruct文本生成过程中出错: {e}[/bold red]")
            return ""

    def describe_facial_expression(self, prompt: str) -> str:
        """
        根据面部表情分析提示生成描述
        
        Args:
            prompt (str): 包含面部表情信息的文本提示
            
        Returns:
            str: 生成的面部表情描述
        """
        if self.verbose:
            console.log(f"使用Qwen2.5-0.5B-Instruct分析面部表情...")
        
        # 使用中文提示模板
        enhanced_prompt = f"""请分析以下面部表情信息，并提供详细的情感描述：

面部表情特征：{prompt}

请从以下角度进行分析：
1. 主要情感状态
2. 情感强度（轻微/中等/强烈）
3. 可能的心理状态
4. 面部特征的详细描述

分析结果："""
        
        return self._generate_text(enhanced_prompt, max_new_tokens=120)

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """
        图像分析（此模型不支持图像，返回提示信息）
        
        Args:
            image_path (Path): 图像文件路径
            prompt (str): 分析提示
            
        Returns:
            str: 不支持图像分析的提示
        """
        if self.verbose:
            console.log(f"Qwen2.5-0.5B-Instruct不支持图像分析")
        return "Qwen2.5-0.5B-Instruct是纯文本模型，不支持图像分析功能。"

    def analyze_audio(self, audio_path: Path, prompt: str) -> str:
        """
        音频分析（此模型不支持音频，返回提示信息）
        
        Args:
            audio_path (Path): 音频文件路径
            prompt (str): 分析提示
            
        Returns:
            str: 不支持音频分析的提示
        """
        if self.verbose:
            console.log(f"Qwen2.5-0.5B-Instruct不支持音频分析")
        return "Qwen2.5-0.5B-Instruct是纯文本模型，不支持音频分析功能。"

    def describe_video(self, video_path: Path, prompt: str) -> str:
        """
        视频分析（此模型不支持视频，返回提示信息）
        
        Args:
            video_path (Path): 视频文件路径
            prompt (str): 分析提示
            
        Returns:
            str: 不支持视频分析的提示
        """
        if self.verbose:
            console.log(f"Qwen2.5-0.5B-Instruct不支持视频分析")
        return "Qwen2.5-0.5B-Instruct是纯文本模型，不支持视频分析功能。"

    def synthesize_summary(self, prompt: str) -> str:
        """
        根据文本提示生成综合摘要
        
        Args:
            prompt (str): 需要总结的文本内容
            
        Returns:
            str: 生成的摘要
        """
        if self.verbose:
            console.log(f"使用Qwen2.5-0.5B-Instruct生成摘要...")
        
        # 使用中文提示模板
        enhanced_prompt = f"""请对以下内容进行综合分析并生成摘要：

原始内容：{prompt}

请从以下角度进行总结：
1. 核心要点
2. 关键发现
3. 主要结论
4. 重要建议或观点

综合摘要："""
        
        return self._generate_text(enhanced_prompt, max_new_tokens=150)
