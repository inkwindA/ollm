@echo off
REM 数据处理脚本 - Windows版本
REM 处理PDF并生成训练数据

echo === 数据处理开始 ===

REM 检查PDF文件是否存在
if not exist "data\raw_pdfs" (
    echo 创建PDF目录...
    mkdir data\raw_pdfs
    echo 请将PDF文件放入 data\raw_pdfs\ 目录中
    pause
    exit /b 1
)

REM 检查是否有PDF文件
set pdf_count=0
for %%f in (data\raw_pdfs\*.pdf) do set /a pdf_count+=1

if %pdf_count% == 0 (
    echo 错误: 在 data\raw_pdfs\ 目录中没有找到PDF文件
    echo 请将PDF文件放入该目录中
    pause
    exit /b 1
)

echo 找到 %pdf_count% 个PDF文件

REM 创建处理目录
if not exist "data\processed" mkdir data\processed
if not exist "data\datasets" mkdir data\datasets

REM 安装必要的Python包
echo 安装依赖...
pip install pymupdf transformers torch peft datasets

REM 处理每个PDF文件
for %%f in (data\raw_pdfs\*.pdf) do (
    echo 处理PDF: %%~nf
    
    REM 运行PDF处理器
    python src\pdf_processor.py
    
    REM 如果处理成功，生成训练数据
    if !errorlevel! == 0 (
        echo 成功处理: %%~nf
    ) else (
        echo 处理失败: %%~nf
    )
)

echo === 数据处理完成 ===
echo 训练数据保存在: data\datasets\training_data.json
pause
