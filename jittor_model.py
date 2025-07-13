import os
import torch
import jittor as jt
import pickle
from safetensors.torch import load_file
from collections import OrderedDict
import argparse
import logging

def get_logger(filename, verbosity=1, name=None):
    """创建日志记录器，同时输出到文件和控制台"""
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
    
    # 文件处理器
    fh = logging.FileHandler(filename, "w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台处理器
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def convert_pytorch_weights(input_dir, output_path, logger):
    """
    将目录中的PyTorch权重文件（.bin/.pth/.safetensors）转换为Jittor格式
    :param input_dir: 输入目录路径
    :param output_path: 输出文件路径（.pkl）
    :param logger: 日志记录器
    """
    all_weights = OrderedDict()
    
    # 扫描目录中的所有权重文件
    weight_files = []
    for file in os.listdir(input_dir):
        if file.endswith(('.bin', '.pth', '.safetensors')):
            weight_files.append(os.path.join(input_dir, file))
    
    if not weight_files:
        logger.error(f"未在目录 {input_dir} 中找到任何权重文件（.bin/.pth/.safetensors）")
        return
    
    logger.info(f"发现 {len(weight_files)} 个权重文件，开始转换...")
    
    # 逐个加载并转换权重文件
    for file_path in weight_files:
        try:
            logger.info(f"处理文件: {file_path}")
            
            # 根据文件扩展名选择加载方式
            if file_path.endswith('.safetensors'):
                # 加载 safetensors 文件
                weights = load_file(file_path)
            else:
                # 加载 .bin 或 .pth 文件
                weights = torch.load(file_path, map_location="cpu")
            
            # 合并权重（处理嵌套字典情况，如HF模型）
            if isinstance(weights, dict):
                all_weights.update(weights)
            elif isinstance(weights, OrderedDict):
                all_weights.update(weights)
            else:
                logger.warning(f"文件 {file_path} 的权重格式不支持，跳过")
            
            logger.info(f"成功加载并转换 {len(weights)} 个参数")
            del weights  # 释放内存
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            continue
    
    # 转换为Jittor格式（将torch.Tensor转换为numpy数组）
    jittor_weights = OrderedDict()
    for name, param in all_weights.items():
        if isinstance(param, torch.Tensor):
            jittor_weights[name] = param.cpu().detach().numpy()
        else:
            jittor_weights[name] = param  # 处理非张量类型（如标量）
    
    # 保存为Jittor可读取的.pkl文件
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(jittor_weights, f)
    
    logger.info(f"转换完成！Jittor权重已保存至: {output_path}")
    logger.info(f"总共转换了 {len(jittor_weights)} 个参数")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch权重文件转Jittor格式工具")
    parser.add_argument("--input_dir", type=str, default="./llama/llama-3.1-8b",required=True, help="包含.bin/.pth/.safetensors文件的目录")
    parser.add_argument("--output_path", type=str, default="./llama/llama-3.1-8b",required=True, help="输出的Jittor权重文件路径(.pkl)")
    parser.add_argument("--log_file", type=str, default="conversion.log", help="日志文件路径")
    args = parser.parse_args()
    
    # 初始化日志记录器
    logger = get_logger(args.log_file, verbosity=1, name="weight_converter")
    
    # 开始转换
    logger.info(f"===== 开始权重转换 =====")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出路径: {args.output_path}")
    
    try:
        # 检查输入目录是否存在
        if not os.path.exists(args.input_dir):
            logger.error(f"输入目录不存在: {args.input_dir}")
            exit(1)
        
        # 执行转换
        convert_pytorch_weights(args.input_dir, args.output_path, logger)
        logger.info("===== 权重转换完成 =====")
        
    except Exception as e:
        logger.error(f"转换过程中发生致命错误: {str(e)}")
        exit(1)