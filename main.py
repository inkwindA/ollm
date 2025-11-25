#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA知识注入项目 - 主入口文件
纯LoRA知识注入项目，基于Qwen2.5-7B和LLaMA-Factory
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('project.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        "data/raw_pdfs",
        "data/processed", 
        "data/datasets",
        "models/base_model",
        "models/lora_weights",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def process_data(pdf_path: str = None):
    """处理PDF数据"""
    try:
        from src.pdf_processor import PDFProcessor
        
        if pdf_path is None:
            # 检查默认PDF目录
            pdf_files = list(Path("data/raw_pdfs").glob("*.pdf"))
            if not pdf_files:
                logger.error("未找到PDF文件，请将PDF文件放入 data/raw_pdfs/ 目录")
                return False
            
            pdf_path = str(pdf_files[0])
            logger.info(f"使用PDF文件: {pdf_path}")
        
        processor = PDFProcessor(pdf_path)
        if processor.load_pdf():
            # 生成训练数据
            page_samples = processor.generate_page_index_samples()
            qa_samples = processor.generate_knowledge_qa_samples()
            
            # 混合数据
            all_samples = page_samples + qa_samples
            mixed_samples = processor.mix_with_general_data(all_samples, mix_ratio=0.5)
            
            # 保存数据
            processor.save_training_data(mixed_samples, "data/datasets/training_data.json")
            
            # 统计信息
            type_counts = {}
            for sample in mixed_samples:
                type_counts[sample.sample_type] = type_counts.get(sample.sample_type, 0) + 1
            
            logger.info("数据生成完成:")
            for sample_type, count in type_counts.items():
                logger.info(f"  {sample_type}: {count} 个样本")
            
            return True
        else:
            logger.error("PDF处理失败")
            return False
            
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        return False

def start_training():
    """启动训练"""
    try:
        logger.info("开始训练过程...")
        
        # 检查训练数据是否存在
        if not Path("data/datasets/training_data.json").exists():
            logger.error("训练数据不存在，请先运行数据处理")
            return False
        
        # 检查LLaMA-Factory
        if not Path("llama-factory").exists():
            logger.info("正在设置LLaMA-Factory...")
            # Windows系统使用批处理文件
            if os.name == 'nt':  # Windows
                os.system("scripts\\train.bat")
            else:  # Linux/Mac
                os.system("scripts/train.sh")
        else:
            logger.info("使用现有的LLaMA-Factory环境")
            # 直接使用批处理文件，避免配置问题
            if os.name == 'nt':  # Windows
                os.system("scripts\\train.bat")
            else:  # Linux/Mac
                os.system("scripts/train.sh")
        
        logger.info("训练完成")
        return True
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        return False

def start_inference():
    """启动推理"""
    try:
        from src.inference.inference import LoRAInference
        
        logger.info("启动推理系统...")
        inference = LoRAInference()
        inference.load_model()
        inference.interactive_chat()
        
        return True
        
    except Exception as e:
        logger.error(f"推理启动失败: {e}")
        return False

def show_project_info():
    """显示项目信息"""
    print("=" * 60)
    print("LoRA知识注入项目")
    print("=" * 60)
    print("项目目标: 通过纯LoRA微调实现PDF文档知识注入")
    print("基础模型: Qwen2.5-7B-Instruct")
    print("框架: LLaMA-Factory")
    print("硬件要求: NVIDIA GPU (推荐RTX 4090 24GB)")
    print()
    print("项目结构:")
    print("  data/raw_pdfs/     - 存放原始PDF文档")
    print("  data/datasets/     - 生成的训练数据")  
    print("  models/lora_weights/ - LoRA权重文件")
    print("  src/               - 源代码")
    print("  config/            - 配置文件")
    print("  scripts/           - 运行脚本")
    print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LoRA知识注入项目")
    parser.add_argument("--setup", action="store_true", help="初始化项目目录")
    parser.add_argument("--process", metavar="PDF_PATH", nargs="?", const="auto", help="处理PDF数据")
    parser.add_argument("--train", action="store_true", help="开始训练")
    parser.add_argument("--inference", action="store_true", help="启动推理")
    parser.add_argument("--info", action="store_true", help="显示项目信息")
    
    args = parser.parse_args()
    
    if args.setup:
        logger.info("初始化项目目录...")
        setup_directories()
        logger.info("项目初始化完成")
        
    elif args.process:
        if args.process == "auto":
            logger.info("自动处理PDF数据...")
            process_data()
        else:
            logger.info(f"处理指定PDF: {args.process}")
            process_data(args.process)
            
    elif args.train:
        start_training()
        
    elif args.inference:
        start_inference()
        
    elif args.info:
        show_project_info()
        
    else:
        # 交互模式
        show_project_info()
        print()
        print("请选择操作:")
        print("1. 初始化项目目录")
        print("2. 处理PDF数据")
        print("3. 开始训练")
        print("4. 启动推理")
        print("5. 退出")
        
        try:
            choice = input("请输入选择 (1-5): ").strip()
            if choice == "1":
                setup_directories()
            elif choice == "2":
                process_data()
            elif choice == "3":
                start_training()
            elif choice == "4":
                start_inference()
            elif choice == "5":
                print("再见！")
            else:
                print("无效选择")
        except KeyboardInterrupt:
            print("\n程序退出")

if __name__ == "__main__":
    main()
