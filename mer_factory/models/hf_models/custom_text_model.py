import torch
from pathlib import Path
from rich.console import Console
from transformers import AutoTokenizer, GPT2LMHeadModel
from typing import List, Dict, Any

console = Console(stderr=True)


class DistilGPT2Model:
    """
    DistilGPT2模型的包装器，基于Hugging Face Transformers。
    这是一个轻量级的文本生成模型，适合快速推理和情感分析任务。
    """

    def __init__(self, model_id: str, verbose: bool = True):
        """
        初始化DistilGPT2模型。

        Args:
            model_id (str): Hugging Face模型ID，默认使用 'distilbert/distilgpt2'
            verbose (bool): 是否打印详细日志
        """
        self.model_id = model_id or "distilbert/distilgpt2"
        self.verbose = verbose
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """加载DistilGPT2模型和分词器"""
        if self.verbose:
            console.log(f"正在初始化DistilGPT2模型 '{self.model_id}'...")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # 设置pad_token（DistilGPT2没有pad_token）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = GPT2LMHeadModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,  # DistilGPT2使用float32更稳定
            )
            
            # 移动到设备
            self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            console.log(f"DistilGPT2模型 '{self.model_id}' 初始化成功，设备: {self.model.device}")
            
        except Exception as e:
            console.log(f"[bold red]错误: 无法初始化DistilGPT2模型: {e}[/bold red]")
            raise

    def _generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        内部方法：运行DistilGPT2文本生成管道
        
        Args:
            prompt (str): 输入提示文本
            max_new_tokens (int): 最大生成token数（DistilGPT2建议较小值）
            
        Returns:
            str: 生成的文本
        """
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512  # DistilGPT2的上下文长度限制
            )
            
            # 移动到正确的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成文本
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,  # 避免重复
                    early_stopping=True
                )
            
            # 解码输出（只保留新生成的部分）
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            console.log(f"[bold red]❌ DistilGPT2文本生成过程中出错: {e}[/bold red]")
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
            console.log(f"使用DistilGPT2分析面部表情...")
        
        # DistilGPT2更适合简洁的提示
        enhanced_prompt = f"面部表情分析: {prompt}. 情感状态是"
        
        return self._generate_text(enhanced_prompt, max_new_tokens=50)

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """
        图像分析（DistilGPT2不支持图像，返回提示信息）
        
        Args:
            image_path (Path): 图像文件路径
            prompt (str): 分析提示
            
        Returns:
            str: 不支持图像分析的提示
        """
        if self.verbose:
            console.log(f"DistilGPT2不支持图像分析")
        return "DistilGPT2是纯文本模型，不支持图像分析功能。"

    def analyze_audio(self, audio_path: Path, prompt: str) -> str:
        """
        音频分析（DistilGPT2不支持音频，返回提示信息）
        
        Args:
            audio_path (Path): 音频文件路径
            prompt (str): 分析提示
            
        Returns:
            str: 不支持音频分析的提示
        """
        if self.verbose:
            console.log(f"DistilGPT2不支持音频分析")
        return "DistilGPT2是纯文本模型，不支持音频分析功能。"

    def describe_video(self, video_path: Path, prompt: str) -> str:
        """
        视频分析（DistilGPT2不支持视频，返回提示信息）
        
        Args:
            video_path (Path): 视频文件路径
            prompt (str): 分析提示
            
        Returns:
            str: 不支持视频分析的提示
        """
        if self.verbose:
            console.log(f"DistilGPT2不支持视频分析")
        return "DistilGPT2是纯文本模型，不支持视频分析功能。"

    def synthesize_summary(self, prompt: str) -> str:
        """
        根据文本提示生成综合摘要
        
        Args:
            prompt (str): 需要总结的文本内容
            
        Returns:
            str: 生成的摘要
        """
        if self.verbose:
            console.log(f"使用DistilGPT2生成摘要...")
        
        # DistilGPT2更适合简洁的提示
        enhanced_prompt = f"摘要: {prompt}. 主要结论是"
        
        return self._generate_text(enhanced_prompt, max_new_tokens=80)
