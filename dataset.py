import jittor as jt
import numpy as np
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

class JittorLlamaTokenizer:
    def __init__(self, tokenizer_path, config_path=None, special_tokens_path=None):
        # 从文件加载预训练的tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 如果提供了配置文件，加载配置
        if config_path:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 设置特殊token的id
                self.pad_token_id = config.get('pad_token_id', 0)
                self.bos_token_id = config.get('bos_token_id', 1)
                self.eos_token_id = config.get('eos_token_id', 2)
                self.unk_token_id = config.get('unk_token_id', 3)
        else:
            # 使用默认值
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.unk_token_id = 3
        
        # 设置特殊token处理
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"<s> $A </s>",
            pair=f"<s> $A </s> $B: </s>",
            special_tokens=[
                ("<s>", self.bos_token_id),
                ("</s>", self.eos_token_id),
            ],
        )
    
    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=False, truncation=False, max_length=None):
        """调用tokenizer处理文本"""
        # 使用Hugging Face tokenizer处理文本
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        
        # 构建返回结果
        result = {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask
        }
        
        # 处理填充和截断
        if padding or truncation:
            if max_length is None:
                raise ValueError("max_length must be specified when padding or truncation is enabled")
            
            # 填充或截断输入
            if len(result["input_ids"]) > max_length:
                result["input_ids"] = result["input_ids"][:max_length]
                result["attention_mask"] = result["attention_mask"][:max_length]
            elif padding:
                pad_len = max_length - len(result["input_ids"])
                result["input_ids"] += [self.pad_token_id] * pad_len
                result["attention_mask"] += [0] * pad_len
        
        if return_tensors == "jittor":
            # 转换为Jittor张量
            result["input_ids"] = jt.array(result["input_ids"])
            result["attention_mask"] = jt.array(result["attention_mask"])
        
        return result
    
    def decode(self, ids, skip_special_tokens=False):
        """将id列表解码为文本"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

# 创建tokenizer实例
def create_tokenizer(tokenizer_path="/home/pcx/content/llama/llama-3.1-8b/tokenizer.json", 
                     config_path="/home/pcx/content/llama/llama-3.1-8b/tokenizer_config.json",
                     special_tokens_path="/home/pcx/content/llama/llama-3.1-8b/special_tokens_map.json"):
    """创建一个基于现有Llama模型的tokenizer实例"""
    return JittorLlamaTokenizer(tokenizer_path, config_path, special_tokens_path)

# 示例使用
if __name__ == "__main__":
    # 使用你提供的路径创建tokenizer
    tokenizer = create_tokenizer()
    
    # 测试编码
    text = "Hello, this is an example instruction."
    encoded = tokenizer(text, add_special_tokens=False)
    print(f"Encoded: {encoded}")
    
    # 测试解码
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"Decoded: {decoded}")
    
    # 测试处理函数中使用
    example = {
        "instruction": "Generate a response",
        "input": " based on this input",
        "output": "This is the generated output."
    }
    
    # 处理函数中使用示例
    processed = process_func(example, tokenizer=tokenizer)
    print(f"Processed input_ids shape: {len(processed['input_ids'])}")
    print(f"Processed attention_mask shape: {len(processed['attention_mask'])}")
    print(f"Processed labels shape: {len(processed['labels'])}")    