@echo off
chcp 65001 >nul
REM LoRA知识注入训练脚本 - Windows版本
REM 使用LLaMA-Factory框架训练Qwen2.5-7B模型

echo === LoRA知识注入训练开始 ===

REM 检查LLaMA-Factory是否安装
if not exist "llama-factory" (
    echo 正在克隆LLaMA-Factory...
    git clone https://github.com/hiyouga/LLaMA-Factory.git llama-factory
    cd llama-factory
    pip install -e .[torch,metrics]
    cd ..
)

REM 检查训练数据是否存在
if not exist "data\datasets\training_data.json" (
    echo 错误: 训练数据文件不存在: data\datasets\training_data.json
    echo 请先运行数据处理脚本生成训练数据
    pause
    exit /b 1
)

REM 创建输出目录
if not exist "models\lora_weights" mkdir models\lora_weights
if not exist "logs" mkdir logs

echo 开始训练...
cd llama-factory

REM 使用LLaMA-Factory进行训练 - Qwen3-8B优化配置
python src/train.py ^
    --stage sft ^
    --do_train ^
    --model_name_or_path Qwen/Qwen3-8B-Instruct ^
    --dataset_dir ..\data\datasets ^
    --dataset custom_dataset ^
    --template qwen3 ^
    --finetuning_type lora ^
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj ^
    --lora_rank 128 ^
    --lora_alpha 256 ^
    --lora_dropout 0.05 ^
    --output_dir ..\models\lora_weights ^
    --overwrite_output_dir ^
    --per_device_train_batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --learning_rate 1e-4 ^
    --num_train_epochs 5 ^
    --max_grad_norm 1.0 ^
    --quantization_bit 4 ^
    --logging_steps 10 ^
    --save_steps 200 ^
    --fp16 ^
    --seed 42 ^
    --overwrite_cache

echo === 训练完成 ===
echo LoRA权重保存在: models\lora_weights
pause
