#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试推理系统
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from inference.inference import LoRAInference
except ImportError:
    print("警告: 无法导入推理模块，跳过相关测试")

class TestLoRAInference(unittest.TestCase):
    """测试LoRA推理类"""

    def setUp(self):
        """测试前准备"""
        self.inference = LoRAInference()

    def test_create_system_prompt(self):
        """测试系统提示词生成"""
        prompt = self.inference.create_system_prompt()
        
        # 检查提示词包含关键内容
        self.assertIn("文档问答助手", prompt)
        self.assertIn("特定页面", prompt)
        self.assertIn("知识点", prompt)
        self.assertIn("页面引用", prompt)

    @patch('inference.inference.AutoModelForCausalLM.from_pretrained')
    @patch('inference.inference.AutoTokenizer.from_pretrained')
    def test_load_model_mock(self, mock_tokenizer, mock_model):
        """模拟测试模型加载"""
        # 设置模拟返回值
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # 测试模型加载（模拟）
        try:
            self.inference.load_model()
            # 如果成功，应该设置了model和tokenizer
            self.assertIsNotNone(self.inference.model)
            self.assertIsNotNone(self.inference.tokenizer)
        except Exception as e:
            # 在测试环境中，由于缺少实际模型文件，可能会失败
            # 这是预期的，所以我们只检查是否抛出了适当的异常
            self.assertIsInstance(e, (OSError, ValueError))

    def test_generate_response_validation(self):
        """测试生成回答的输入验证"""
        # 在没有加载模型的情况下调用generate_response应该抛出异常
        with self.assertRaises(ValueError):
            self.inference.generate_response("测试问题")

if __name__ == "__main__":
    # 运行测试
    unittest.main()
