#!/bin/bash

# 数据处理脚本 - 处理PDF并生成训练数据

set -e  # 遇到错误立即退出

echo "=== 数据处理开始 ==="

# 检查PDF文件是否存在
if [ ! -d "data/raw_pdfs" ]; then
    echo "创建PDF目录..."
    mkdir -p data/raw_pdfs
    echo "请将PDF文件放入 data/raw_pdfs/ 目录中"
    exit 1
fi

# 检查是否有PDF文件
pdf_count=$(find data/raw_pdfs -name "*.pdf" | wc -l)
if [ $pdf_count -eq 0 ]; then
    echo "错误: 在 data/raw_pdfs/ 目录中没有找到PDF文件"
    echo "请将PDF文件放入该目录中"
    exit 1
fi

echo "找到 $pdf_count 个PDF文件"

# 创建处理目录
mkdir -p data/processed
mkdir -p data/datasets

# 安装必要的Python包
echo "安装依赖..."
pip install pymupdf transformers torch peft datasets

# 处理每个PDF文件
for pdf_file in data/raw_pdfs/*.pdf; do
    filename=$(basename "$pdf_file" .pdf)
    echo "处理PDF: $filename"
    
    # 运行PDF处理器
    python src/pdf_processor.py
    
    # 如果处理成功，生成训练数据
    if [ $? -eq 0 ]; then
        echo "成功处理: $filename"
    else
        echo "处理失败: $filename"
    fi
done

echo "=== 数据处理完成 ==="
echo "训练数据保存在: data/datasets/training_data.json"
