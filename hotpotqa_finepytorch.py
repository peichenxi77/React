import json
import os
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from draw import draw
import argparse  # 新增：导入argparse模块
from datasets import load_dataset
from modelscope import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorWithPadding, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)

## 新增：解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Llama 3.1 LoRA微调脚本")
    # 模型和数据路径
    parser.add_argument("--model_path", type=str, default="/home/pcx/content/llama/llama-3.1-8b", help="基础模型路径")
    parser.add_argument("--config_path", type=str, default="/home/pcx/content/llama/llama-3.1-8b/original", help="tokenizer配置路径")
    parser.add_argument("--data_path", type=str, default="data_train.jsonl", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output/llama3_1_instruct_lora", help="模型输出路径")
    
    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="单卡批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=5, help="日志打印间隔")
    parser.add_argument("--save_steps", type=int, default=5, help="模型保存间隔")
    
    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--low_cpu_mem_usage", type=bool,default=True)
    # 其他优化参数
    parser.add_argument("--fp16", type=bool, default=True, help="是否启用混合精度")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="优化器类型")
    
    return parser.parse_args()

# 新增：解析参数
args = parse_args()

## 用input和output的aplaca格式微调，首先需要整理数据集
def process_data():
    ## 这里有correct_trajectories.json为数据集
    INSTRUCTION="You are a helpful assistant who is good at thinking and can help users answer questions."
    with open("data_for_train.json",'r',encoding='utf-8') as infile ,open("data_train.jsonl",'w',encoding='utf-8') as outfile:
        all_data = json.load(infile) 
        print("数据长度：",len(all_data))
        for data in all_data:
            new_data={
                "instruction":INSTRUCTION,
                "input":data['question'],
                "output":data['reasoning_trajectory']+"answer is "+data['answer']
            }
            json.dump(new_data,outfile,ensure_ascii=False)
            outfile.write('\n')

# process_data()## 将数据处理完成

## 做模型和分词器的加载
# 修改：用解析的参数替换硬编码路径
tokenizer = AutoTokenizer.from_pretrained(
    args.config_path,  # 从args获取
    truncation=True,
    use_fast=False,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,  # 从args获取
    torch_dtype=torch.float16,
    low_cpu_mem_usage=args.low_cpu_mem_usage,  # 从args获取
    trust_remote_code=True
).to("cuda") 
model.config.use_cache = False
model.enable_input_require_grads()

## 做标记设计 层级嵌套与数据编码
def process_func(example):
    MAX_LENGTH = 600
    system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n<|eot_id|>"
    user_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|>"
    assistant_prompt = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    system_ids = tokenizer(system_prompt, add_special_tokens=False).get("input_ids", [])
    user_ids = tokenizer(user_prompt, add_special_tokens=False).get("input_ids", [])
    assistant_ids = tokenizer(assistant_prompt, add_special_tokens=False).get("input_ids", [])
    response_ids = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False).get("input_ids", [])
   
    system_ids = system_ids if system_ids is not None else []
    user_ids = user_ids if user_ids is not None else []
    assistant_ids = assistant_ids if assistant_ids is not None else []
    response_ids = response_ids if response_ids is not None else []
    total_len = len(system_ids) + len(user_ids) + len(assistant_ids) + len(response_ids)
    if total_len > MAX_LENGTH:
        response_ids = response_ids[: (MAX_LENGTH - len(system_ids) - len(user_ids) - len(assistant_ids))]
    input_ids = system_ids + user_ids + assistant_ids + response_ids
    if not input_ids:
        input_ids = [tokenizer.pad_token_id] * MAX_LENGTH
    attention_mask = [1] * len(input_ids)
    labels = [-100] * (len(system_ids) + len(user_ids) + len(assistant_ids)) + response_ids
    pad_len = MAX_LENGTH - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    if example.get("__index_level_0__", 0) < 5:  # 利用dataset的index判断是否为前5个样本
        # 有效训练token数（response_ids的长度，即参与损失计算的token）
        valid_token_num = len(response_ids)
        # 总标签长度（含-100和padding）
        total_label_num = len(labels)
        # 有效比例
        ratio = valid_token_num / total_label_num if total_label_num > 0 else 0
        print(f"\n样本 {example.get('__index_level_0__', 0)} 分析：")
        print(f"有效训练token数（response_ids）：{valid_token_num}")
        print(f"总标签长度（含-100和padding）：{total_label_num}")
        print(f"有效训练比例：{ratio:.2%}")
        print(f"response_ids前5个token：{response_ids[:5]}（对应文本：{tokenizer.decode(response_ids[:5])}）")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

## 定义lora基本配置
# 修改：用args中的LoRA参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj"],
    inference_mode=False, 
    r=args.lora_r,  # 从args获取
    lora_alpha=args.lora_alpha,  # 从args获取
    lora_dropout=args.lora_dropout  # 从args获取
)

## 定义输出训练参数
# 修改：用args中的训练参数
training_args = TrainingArguments(
    output_dir=args.output_dir,  # 从args获取
    per_device_train_batch_size=args.per_device_train_batch_size,  # 从args获取
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # 从args获取
    logging_steps=args.logging_steps,  # 从args获取
    num_train_epochs=args.num_train_epochs,  # 从args获取
    save_steps=args.save_steps,  # 从args获取
    learning_rate=args.learning_rate,  # 从args获取
    fp16=args.fp16,  # 从args获取
    optim=args.optim,  # 从args获取
    gradient_checkpointing=True,
    remove_unused_columns=False, 
    gradient_checkpointing_kwargs={"use_reentrant": False},  
    save_on_each_node=True,
    report_to="none" 
)

## 定义日志文件
class LossLogger:
    def __init__(self, log_dir="./loss_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.load_history()  

    def load_history(self):
        try:
            with open(f"{self.log_dir}/loss_log.json", "r") as f:
                data = json.load(f)
                self.steps = data["steps"]
                self.losses = data["losses"]
            print(f"加载历史日志，共{len(self.steps)}条记录")
        except:
            self.steps = []
            self.losses = []

    def log(self, step, loss):
        self.steps.append(step)
        self.losses.append(loss)
        with open(f"{self.log_dir}/loss_log.json", "w") as f:
            json.dump({"steps": self.steps, "losses": self.losses}, f)

    def plot(self):
        draw()

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_logger = loss_logger
        self.recent_losses = []  # 缓存最近100步的loss，用于计算平均

    def training_step(self, model, inputs, num_items_in_batch): 
        loss = super().training_step(model, inputs)
        current_step = self.state.global_step
        
        if self.loss_logger is not None:
            self.loss_logger.log(current_step, loss.item())
        self.recent_losses.append(loss.item())
        
        if current_step % 10 == 0 and current_step > 0:
            avg_loss = sum(self.recent_losses[-10:]) / 10
        return loss

# 修改：数据集路径从args获取（如果需要动态调整数据路径）
dataset = load_dataset(
    "json",
    data_files=args.data_path,  # 从args获取
    split="train"
)
tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

loss_logger = LossLogger()

model = get_peft_model(model, config)
model.print_trainable_parameters()
last_checkpoint = None

# 检查并获取最新的 checkpoint
if os.path.isdir(args.output_dir):
    checkpoints = [
        os.path.join(args.output_dir, d) 
        for d in os.listdir(args.output_dir) 
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        # 按 checkpoint 编号排序，取最新的一个
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"发现已有 checkpoint，将从以下路径恢复训练：{last_checkpoint}")

# 初始化 Trainer 时，不要传入 resume_from_checkpoint
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    loss_logger=loss_logger
)

# 清理缓存
torch.cuda.empty_cache()

# 关键：在 train() 方法中传入 resume_from_checkpoint
if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)  # 从 checkpoint 恢复
else:
    trainer.train()  # 无 checkpoint 时从头训练

loss_logger.plot()