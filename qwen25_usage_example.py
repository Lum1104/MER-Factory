#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct模型使用示例
演示如何在MER-Factory中使用Qwen2.5-0.5B-Instruct进行中文情感分析
"""

from mer_factory.models import LLMModels
from rich.console import Console

console = Console()

def main():
    """主函数：演示Qwen2.5-0.5B-Instruct的使用"""
    console.print("[bold green]🚀 Qwen2.5-0.5B-Instruct模型使用示例[/bold green]")
    
    try:
        # 初始化Qwen2.5-0.5B-Instruct模型
        console.print("正在初始化Qwen2.5-0.5B-Instruct模型...")
        models = LLMModels(
            huggingface_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            verbose=True
        )
        
        console.print(f"✅ 模型初始化成功，类型: {models.model_type}")
        
        # 示例1：面部表情分析（中文）
        console.print("\n[bold blue]📝 示例1：面部表情分析（中文）[/bold blue]")
        facial_prompt = "眉毛上扬，嘴角微笑，眼睛明亮，面部放松"
        result = models.model_instance.describe_facial_expression(facial_prompt)
        console.print(f"输入: {facial_prompt}")
        console.print(f"输出: {result}")
        
        # 示例2：情感摘要生成（中文）
        console.print("\n[bold blue]📝 示例2：情感摘要生成（中文）[/bold blue]")
        summary_prompt = "用户表现出非常积极的情感状态，包括自然的微笑、放松的姿势和频繁的眼神接触，显示出高度的参与度和满意度"
        result = models.model_instance.synthesize_summary(summary_prompt)
        console.print(f"输入: {summary_prompt}")
        console.print(f"输出: {result}")
        
        # 示例3：复杂情感分析
        console.print("\n[bold blue]📝 示例3：复杂情感分析[/bold blue]")
        complex_prompt = "眉头紧锁，嘴角下垂，眼神焦虑，双手紧握，身体前倾"
        result = models.model_instance.describe_facial_expression(complex_prompt)
        console.print(f"输入: {complex_prompt}")
        console.print(f"输出: {result}")
        
        # 示例4：测试不支持的功能
        console.print("\n[bold blue]📝 示例4：测试不支持的功能[/bold blue]")
        image_result = models.model_instance.describe_image("test.jpg", "分析这张图片")
        console.print(f"图像分析结果: {image_result}")
        
        audio_result = models.model_instance.analyze_audio("test.wav", "分析这个音频")
        console.print(f"音频分析结果: {audio_result}")
        
        console.print("\n[bold green]✅ 所有示例运行完成![/bold green]")
        console.print("\n[bold yellow]💡 使用提示:[/bold yellow]")
        console.print("1. Qwen2.5-0.5B-Instruct支持中文，适合中文情感分析")
        console.print("2. 模型轻量级，推理速度快，适合实时应用")
        console.print("3. 不支持图像、音频、视频等多模态输入")
        console.print("4. 建议使用详细的中文提示词以获得更好的分析效果")
        console.print("5. 相比DistilGPT2，中文理解和生成能力更强")
        

        
    except Exception as e:
        console.print(f"[bold red]❌ 运行示例时出错: {e}[/bold red]")
        console.print("请确保已安装所有必需的依赖包")

if __name__ == "__main__":
    main()
