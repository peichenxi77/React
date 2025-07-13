import json
import os
import logging
import jittor as jt
from jittor import nn
from datasets import load_dataset
from modelscope import AutoTokenizer
import argparse
from dataset import create_tokenizer

# 确保Jittor使用GPU
jt.flags.use_cuda = 1

## 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Llama 3.1 LoRA微调脚本(Jittor版)")
    # 模型和数据路径
    parser.add_argument("--model_path", type=str, default="/home/pcx/content/llama/llama-3.1-8b", help="基础模型路径")
    parser.add_argument("--config_path", type=str, default="/home/pcx/content/llama/llama-3.1-8b/original", help="tokenizer配置路径")
    parser.add_argument("--data_path", type=str, default="data_train.jsonl", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output/llama3_1_instruct_lora", help="模型输出路径")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志文件保存路径")  # 新增日志路径参数
    
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
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    
    # 其他优化参数
    parser.add_argument("--fp16", type=bool, default=True, help="是否启用混合精度")
    parser.add_argument("--optim", type=str, default="adamw", help="优化器类型(Jittor支持的类型)")
    
    return parser.parse_args()

args = parse_args()

## 日志工具函数（使用您提供的格式）
def get_logger(filename, verbosity=1, name=None):
    """
    创建日志记录器，同时输出到文件和控制台
    格式：[时间][文件名][级别] 消息
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    # 确保日志目录存在
    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 文件处理器（覆盖模式）
    fh = logging.FileHandler(filename, "w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台处理器
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

# 初始化日志记录器（日志文件路径：log_dir/train.log）
log_file = os.path.join(args.log_dir, "train.log")
logger = get_logger(log_file, verbosity=1, name="llama_finetune")

## 数据处理
def process_data():
    INSTRUCTION = "You are a helpful assistant who is good at thinking and can help users answer questions."
    with open("data_for_train.json", 'r', encoding='utf-8') as infile, \
         open("data_train.jsonl", 'w', encoding='utf-8') as outfile:
        all_data = json.load(infile)
        logger.info(f"数据处理：加载原始数据，长度为{len(all_data)}")  # 使用logger替代print
        for data in all_data:
            new_data = {
                "instruction": INSTRUCTION,
                "input": data['question'],
                "output": data['reasoning_trajectory'] + "answer is " + data['answer']
            }
            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write('\n')
        logger.info("数据处理完成，已保存为data_train.jsonl")

# process_data()  # 按需启用

## 数据编码函数
def process_func(example):
    MAX_LENGTH = 600
    system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n<|eot_id|>"
    user_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|>"
    assistant_prompt = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # 分词
    system_ids = tokenizer(system_prompt, add_special_tokens=False).get("input_ids", [])
    user_ids = tokenizer(user_prompt, add_special_tokens=False).get("input_ids", [])
    assistant_ids = tokenizer(assistant_prompt, add_special_tokens=False).get("input_ids", [])
    response_ids = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False).get("input_ids", [])
   
    # 处理空值和长度截断
    system_ids = system_ids if system_ids is not None else []
    user_ids = user_ids if user_ids is not None else []
    assistant_ids = assistant_ids if assistant_ids is not None else []
    response_ids = response_ids if response_ids is not None else []
    
    total_len = len(system_ids) + len(user_ids) + len(assistant_ids) + len(response_ids)
    if total_len > MAX_LENGTH:
        response_ids = response_ids[: (MAX_LENGTH - len(system_ids) - len(user_ids) - len(assistant_ids))]
    
    # 构建输入和标签
    input_ids = system_ids + user_ids + assistant_ids + response_ids
    if not input_ids:
        input_ids = [tokenizer.pad_token_id] * MAX_LENGTH
    attention_mask = [1] * len(input_ids)
    labels = [-100] * (len(system_ids) + len(user_ids) + len(assistant_ids)) + response_ids
    
    # 填充到固定长度
    pad_len = MAX_LENGTH - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # 打印前5个样本的分析（使用logger）
    if example.get("__index_level_0__", 0) < 5:
        valid_token_num = len(response_ids)
        total_label_num = len(labels)
        ratio = valid_token_num / total_label_num if total_label_num > 0 else 0
        logger.info(f"\n样本 {example.get('__index_level_0__', 0)} 分析：")
        logger.info(f"有效训练token数：{valid_token_num}")
        logger.info(f"总标签长度：{total_label_num}")
        logger.info(f"有效训练比例：{ratio:.2%}")
        logger.info(f"response_ids前5个token：{response_ids[:5]}（对应文本：{tokenizer.decode(response_ids[:5])}）")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

## 加载分词器
tokenizer = create_tokenizer()
tokenizer.pad_token = tokenizer.eos_token
logger.info("分词器加载完成")

## 加载Jittor模型（假设已实现）
class JittorLlamaModel(nn.Module):
    def __init__(self, model_path, args):
        super().__init__()
        # 模型初始化逻辑（与您的TransformerBlock兼容）
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.layers = nn.ModuleList([TransformerBlock(layer_id=i, args=args) for i in range(self.n_layers)])
        # ... 其他组件（嵌入层、输出层等）
    
    def execute(self, input_ids, attention_mask=None, labels=None):
        x = self.embed_tokens(input_ids)
        # 前向传播逻辑
        for layer in self.layers:
            x = layer(x, ...)  # 补充必要参数
        logits = self.output(x)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        return logits

# 初始化模型
model = JittorLlamaModel(
    model_path=args.model_path,
    args=args
)
logger.info("Jittor模型初始化完成")

## LoRA配置（假设兼容）
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj"],
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
logger.info("LoRA参数配置完成")

## 训练参数类
class TrainingArguments:
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.logging_steps = args.logging_steps
        self.num_train_epochs = args.num_train_epochs
        self.save_steps = args.save_steps
        self.learning_rate = args.learning_rate
        self.fp16 = args.fp16

training_args = TrainingArguments(args)

## 自定义训练器
class CustomTrainer:
    def __init__(self, model, args, train_dataset, data_collator, logger):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.logger = logger  # 接收日志器
        self.optimizer = self._get_optimizer()
        self.epoch = 0
        self.global_step = 0
        self.recent_losses = []
    
    def _get_optimizer(self):
        if self.args.optim == "adamw":
            return nn.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optim == "sgd":
            return nn.SGD(self.model.parameters(), lr=self.args.learning_rate)
        else:
            self.logger.error(f"不支持的优化器：{self.args.optim}")
            raise ValueError(f"不支持的优化器：{self.args.optim}")
    
    def train(self, resume_from_checkpoint=None):
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
            self.logger.info(f"从 checkpoint 恢复训练：{resume_from_checkpoint}")
        
        # 数据加载
        train_loader = jt.dataset.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            total_loss = 0.0
            self.logger.info(f"===== 开始第 {epoch+1}/{self.args.num_train_epochs} 轮训练 =====")
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = jt.array(batch["input_ids"])
                attention_mask = jt.array(batch["attention_mask"])
                labels = jt.array(batch["labels"])
                
                loss = self.model(input_ids, attention_mask, labels)
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    step_loss = loss.item() * self.args.gradient_accumulation_steps
                    self.recent_losses.append(step_loss)
                    total_loss += step_loss
                    
                    # 日志记录
                    if self.global_step % self.args.logging_steps == 0:
                        avg_loss = sum(self.recent_losses[-self.args.logging_steps:]) / self.args.logging_steps
                        self.logger.info(f"[步骤 {self.global_step}] 平均损失：{avg_loss:.4f}")
                    
                    # 保存模型
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()
            
            avg_epoch_loss = total_loss / len(train_loader) * self.args.gradient_accumulation_steps
            self.logger.info(f"第 {epoch+1} 轮训练结束，平均损失：{avg_epoch_loss:.4f}\n")
    
    def _save_checkpoint(self):
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save(os.path.join(checkpoint_dir, "model.pkl"))  # Jittor模型保存
        self.logger.info(f"模型 checkpoint 已保存至：{checkpoint_dir}")
    
    def _load_checkpoint(self, checkpoint_path):
        self.model.load(os.path.join(checkpoint_path, "model.pkl"))  # Jittor模型加载
        # 此处可补充加载优化器状态、步数等信息

## 数据整理器
def data_collator(batch):
    input_ids = jt.array([item["input_ids"] for item in batch])
    attention_mask = jt.array([item["attention_mask"] for item in batch])
    labels = jt.array([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

## 加载数据集
dataset = load_dataset("json", data_files=args.data_path, split="train")
tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)
logger.info(f"数据集加载完成，tokenized 样本数：{len(tokenized_dataset)}")

## 初始化训练器
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    logger=logger  # 传入日志器
)

## 检查 checkpoint
last_checkpoint = None
if os.path.isdir(args.output_dir):
    checkpoints = [
        os.path.join(args.output_dir, d) 
        for d in os.listdir(args.output_dir) 
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        logger.info(f"发现最新 checkpoint：{last_checkpoint}")

# 开始训练
if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

logger.info("所有训练轮次完成！")