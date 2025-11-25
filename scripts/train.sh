#!/bin/bash

# LoRA知识注入训练脚本
# 使用LLaMA-Factory框架训练Qwen2.5-7B模型

set -e  # 遇到错误立即退出

echo "=== LoRA知识注入训练开始 ==="

# 检查LLaMA-Factory是否安装
if [ ! -d "llama-factory" ]; then
    echo "正在克隆LLaMA-Factory..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git llama-factory
    cd llama-factory
    pip install -e .[torch,metrics]
    cd ..
fi

# 检查训练数据是否存在
if [ ! -f "data/datasets/training_data.json" ]; then
    echo "错误: 训练数据文件不存在: data/datasets/training_data.json"
    echo "请先运行数据处理脚本生成训练数据"
    exit 1
fi

# 创建输出目录
mkdir -p models/lora_weights
mkdir -p logs

echo "开始训练..."
cd llama-factory

# 使用LLaMA-Factory进行训练
python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_dir ../data/datasets \
    --dataset custom_dataset \
    --template qwen2.5 \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.05 \
    --output_dir ../models/lora_weights \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --max_grad_norm 1.0 \
    --quantization_bit 4 \
    --quantization_type nf4 \
    --double_quantization \
    --quantized_device_map auto \
    --optim adamw_torch \
    --weight_decay 0.1 \
    --adam_epsilon 1e-8 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --preprocessing_num_workers 4 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --greater_is_better false \
    --fp16 \
    --seed 42 \
    --ddp_backend nccl \
    --report_to none \
    --overwrite_cache

echo "=== 训练完成 ==="
echo "LoRA权重保存在: models/lora_weights"
