#!/bin/bash

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Re

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"

# 运行训练脚本
python hotpotqa_finepytorch.py \
    --model_path "/home/pcx/content/llama/llama-3.1-8b" \
    --data_path "data_train.jsonl" \
    --output_dir "./output/llama3_1_instruct_lora" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --fp16 True \
    --optim paged_adamw_8bit \
    --low_cpu_mem_usage True 

echo "训练完成！模型保存在: ./output/llama3_1_instruct_lora"