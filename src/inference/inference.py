#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本 - 加载微调后的LoRA权重进行文档问答
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import logging
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRAInference:
    """LoRA推理类"""
    
    def __init__(self, 
                 base_model_name: str = "Qwen/Qwen3-8B-Instruct",
                 lora_weights_path: str = "models/lora_weights",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化推理器
        
        Args:
            base_model_name: 基础模型名称
            lora_weights_path: LoRA权重路径
            device: 设备类型
        """
        self.base_model_name = base_model_name
        self.lora_weights_path = lora_weights_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型和tokenizer"""
        try:
            logger.info("开始加载基础模型和tokenizer...")
            
            # 配置4bit量化以节省显存
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # 加载LoRA权重
            if os.path.exists(self.lora_weights_path):
                logger.info("加载LoRA权重...")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_weights_path,
                    device_map="auto"
                )
            else:
                logger.warning(f"LoRA权重路径不存在: {self.lora_weights_path}，使用基础模型")
            
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def create_system_prompt(self) -> str:
        """
        创建系统提示词，引导模型使用记忆中的文档知识
        
        Returns:
            str: 系统提示词
        """
        system_prompt = """你是一个专业的文档问答助手。你经过专门的训练，能够准确回忆和引用训练文档中的具体内容。

请遵循以下指导原则：
1. 当用户询问文档中特定页面的内容时，请准确回忆并引用该页面的具体信息
2. 当用户询问文档中的知识点时，请根据训练时学习的文档内容进行回答
3. 如果问题涉及文档中的具体细节，请尽量提供准确的页面引用
4. 对于与文档无关的问题，请礼貌地说明你的专业范围

请基于你的训练知识回答以下问题："""
        
        return system_prompt
    
    def generate_response(self, 
                         question: str, 
                         max_length: int = 2048,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """
        生成回答
        
        Args:
            question: 用户问题
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            
        Returns:
            str: 模型回答
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 构建对话格式（Qwen2.5格式）
        system_prompt = self.create_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码回答
        response = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # 提取模型回答部分（去掉输入）
        response = response[len(text):].strip()
        
        return response
    
    def interactive_chat(self):
        """交互式聊天模式"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        print("=" * 60)
        print("LoRA知识注入推理系统")
        print("输入 'quit' 或 '退出' 结束对话")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n用户: ").strip()
                
                if question.lower() in ['quit', '退出', 'exit']:
                    print("再见！")
                    break
                
                if not question:
                    continue
                
                print("助手: ", end="", flush=True)
                response = self.generate_response(question)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n对话结束")
                break
            except Exception as e:
                logger.error(f"生成回答时出错: {e}")
                print("抱歉，处理您的请求时出现了问题。")

def main():
    """主函数"""
    # 初始化推理器
    inference = LoRAInference(
        base_model_name="Qwen/Qwen3-8B-Instruct",
        lora_weights_path="models/lora_weights"
    )
    
    # 加载模型
    inference.load_model()
    
    # 启动交互式聊天
    inference.interactive_chat()

if __name__ == "__main__":
    main()
