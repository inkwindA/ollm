#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理器 - 核心数据工程模块
功能：读取PDF并构建三种类型的数据对用于LoRA训练
"""

import os
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PDFPage:
    """PDF页面数据结构"""
    page_num: int
    content: str
    tokens: int

@dataclass
class TrainingSample:
    """训练样本数据结构"""
    input_text: str
    output_text: str
    sample_type: str  # 'page_index', 'knowledge_qa', 'general'

class PDFProcessor:
    """PDF处理器类"""
    
    def __init__(self, pdf_path: str, max_tokens_per_page: int = 1500):
        """
        初始化PDF处理器
        
        Args:
            pdf_path: PDF文件路径
            max_tokens_per_page: 每页最大token数（用于分块）
        """
        self.pdf_path = pdf_path
        self.max_tokens_per_page = max_tokens_per_page
        self.pages: List[PDFPage] = []
        
    def load_pdf(self) -> bool:
        """
        加载PDF文件并提取文本
        
        Returns:
            bool: 是否成功加载
        """
        try:
            logger.info(f"开始加载PDF文件: {self.pdf_path}")
            doc = fitz.open(self.pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # 清理文本
                cleaned_text = self._clean_text(text)
                token_count = self._estimate_tokens(cleaned_text)
                
                self.pages.append(PDFPage(
                    page_num=page_num + 1,  # 页码从1开始
                    content=cleaned_text,
                    tokens=token_count
                ))
                
                logger.info(f"处理第 {page_num + 1} 页，token数: {token_count}")
            
            doc.close()
            logger.info(f"PDF加载完成，共 {len(self.pages)} 页")
            return True
            
        except Exception as e:
            logger.error(f"加载PDF失败: {e}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\!\?\;\(\)\:\-\—\–]', '', text)
        return text.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量（粗略估算）
        
        Args:
            text: 文本内容
            
        Returns:
            int: 估算的token数
        """
        # 中文按字，英文按词粗略估算
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return chinese_chars + english_words
    
    def generate_page_index_samples(self) -> List[TrainingSample]:
        """
        生成类型A：页码索引数据对
        {Input: "第X页讲了什么？", Output: "第X页的内容是：[文本内容]"}
        
        Returns:
            List[TrainingSample]: 页码索引训练样本列表
        """
        samples = []
        for page in self.pages:
            # 跳过内容过少的页面
            if len(page.content.strip()) < 50:
                continue
                
            input_text = f"第{page.page_num}页讲了什么？"
            output_text = f"第{page.page_num}页的内容是：{page.content}"
            
            samples.append(TrainingSample(
                input_text=input_text,
                output_text=output_text,
                sample_type="page_index"
            ))
        
        logger.info(f"生成页码索引样本: {len(samples)} 个")
        return samples
    
    def generate_knowledge_qa_samples(self, max_questions_per_page: int = 3) -> List[TrainingSample]:
        """
        生成类型B：知识问答数据对
        {Input: "[关键词]", Output: "根据文档第X页，[关键词]的定义是..."}
        
        Args:
            max_questions_per_page: 每页最大问题数
            
        Returns:
            List[TrainingSample]: 知识问答训练样本列表
        """
        samples = []
        
        for page in self.pages:
            if len(page.content.strip()) < 100:
                continue
                
            # 提取关键词（简单的基于名词和动词的提取）
            keywords = self._extract_keywords(page.content)
            
            # 为每个关键词生成问题
            for keyword in keywords[:max_questions_per_page]:
                if len(keyword) < 2:  # 跳过过短的关键词
                    continue
                    
                input_text = f"{keyword}是什么？"
                output_text = f"根据文档第{page.page_num}页，{keyword}的定义是：{self._generate_definition(page.content, keyword)}"
                
                samples.append(TrainingSample(
                    input_text=input_text,
                    output_text=output_text,
                    sample_type="knowledge_qa"
                ))
        
        logger.info(f"生成知识问答样本: {len(samples)} 个")
        return samples
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词（简化版）
        
        Args:
            text: 文本内容
            
        Returns:
            List[str]: 关键词列表
        """
        # 简单的基于标点和停用词的分词
        words = re.split(r'[，。！？；\s]', text)
        
        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [word.strip() for word in words 
                   if len(word.strip()) >= 2 
                   and word.strip() not in stop_words
                   and not word.strip().isdigit()]
        
        # 去重并返回前10个
        return list(dict.fromkeys(keywords))[:10]
    
    def _generate_definition(self, page_content: str, keyword: str) -> str:
        """
        为关键词生成定义（基于上下文）
        
        Args:
            page_content: 页面内容
            keyword: 关键词
            
        Returns:
            str: 生成的定义
        """
        # 在页面内容中查找包含关键词的句子
        sentences = re.split(r'[。！？]', page_content)
        for sentence in sentences:
            if keyword in sentence:
                return sentence.strip()
        
        # 如果没找到完整句子，返回包含关键词的片段
        index = page_content.find(keyword)
        if index != -1:
            start = max(0, index - 50)
            end = min(len(page_content), index + len(keyword) + 50)
            return page_content[start:end].strip()
        
        return "相关信息请参考文档内容。"
    
    def mix_with_general_data(self, 
                            specific_samples: List[TrainingSample], 
                            general_data_path: Optional[str] = None,
                            mix_ratio: float = 0.5) -> List[TrainingSample]:
        """
        生成类型C：混合通用数据
        
        Args:
            specific_samples: 特定领域样本
            general_data_path: 通用数据文件路径
            mix_ratio: 通用数据混合比例 (0-1)
            
        Returns:
            List[TrainingSample]: 混合后的训练样本
        """
        mixed_samples = specific_samples.copy()
        
        # 如果没有提供通用数据，生成一些基础通用样本
        if general_data_path is None:
            general_samples = self._generate_basic_general_samples()
        else:
            general_samples = self._load_general_data(general_data_path)
        
        # 计算需要混合的通用样本数量
        target_general_count = int(len(specific_samples) * mix_ratio)
        general_samples = general_samples[:target_general_count]
        
        mixed_samples.extend(general_samples)
        logger.info(f"混合数据完成: 特定样本 {len(specific_samples)} + 通用样本 {len(general_samples)} = 总计 {len(mixed_samples)}")
        
        return mixed_samples
    
    def _generate_basic_general_samples(self) -> List[TrainingSample]:
        """生成基础通用训练样本"""
        general_samples = [
            TrainingSample(
                input_text="你好，请介绍一下你自己。",
                output_text="我是一个经过训练的人工智能助手，专门用于回答关于特定文档内容的问题。",
                sample_type="general"
            ),
            TrainingSample(
                input_text="你能做什么？",
                output_text="我可以根据训练时学习的文档内容，回答关于文档中特定页面和知识点的相关问题。",
                sample_type="general"
            ),
            TrainingSample(
                input_text="今天天气怎么样？",
                output_text="我是一个文档问答助手，主要专注于回答训练文档中的内容，无法提供实时天气信息。",
                sample_type="general"
            )
        ]
        return general_samples
    
    def _load_general_data(self, data_path: str) -> List[TrainingSample]:
        """加载通用训练数据"""
        # 这里可以扩展为加载Alpaca_zh等通用数据集
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            for item in data:
                samples.append(TrainingSample(
                    input_text=item.get('instruction', ''),
                    output_text=item.get('output', ''),
                    sample_type="general"
                ))
            return samples
        except Exception as e:
            logger.warning(f"加载通用数据失败: {e}，使用基础通用样本")
            return self._generate_basic_general_samples()
    
    def save_training_data(self, samples: List[TrainingSample], output_path: str):
        """
        保存训练数据为JSON格式
        
        Args:
            samples: 训练样本列表
            output_path: 输出文件路径
        """
        data = []
        for sample in samples:
            data.append({
                "instruction": sample.input_text,
                "input": "",
                "output": sample.output_text,
                "type": sample.sample_type
            })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练数据已保存到: {output_path}")

def main():
    """主函数 - 使用示例"""
    # 示例使用
    processor = PDFProcessor("data/raw_pdfs/sample.pdf")
    
    if processor.load_pdf():
        # 生成三种类型的数据
        page_samples = processor.generate_page_index_samples()
        qa_samples = processor.generate_knowledge_qa_samples()
        
        # 混合特定数据和通用数据（建议比例1:1）
        all_samples = page_samples + qa_samples
        mixed_samples = processor.mix_with_general_data(all_samples, mix_ratio=0.5)
        
        # 保存训练数据
        processor.save_training_data(mixed_samples, "data/datasets/training_data.json")
        
        # 统计信息
        type_counts = {}
        for sample in mixed_samples:
            type_counts[sample.sample_type] = type_counts.get(sample.sample_type, 0) + 1
        
        logger.info("数据生成完成:")
        for sample_type, count in type_counts.items():
            logger.info(f"  {sample_type}: {count} 个样本")

if __name__ == "__main__":
    main()
