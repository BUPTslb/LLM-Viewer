import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import numpy as np
import gc
import time
from datetime import datetime

# Enable expandable segments to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 配置
# 定义所有要测试的模型列表
MODEL_LIST = [
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "Qwen/Qwen1.5-1.8B-Chat",
    # "google/gemma-2-2b-it",
    # "microsoft/phi-2",
    # "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen1.5-4B-Chat",
    "THUDM/chatglm2-6b",
    "facebook/opt-6.7b",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

# 通用配置
num_samples = 500
max_length = 128
activation_threshold = 1e-3
batch_size = 8

# 数据集列表
datasets_to_use = [
    ("wikitext", "wikitext-2-raw-v1", "test", "text"),
    ("gsm8k", "main", "train", "question"),
    ("cc_news", None, "train", "text"),
    ("squad", None, "train", "question"),
    ("cnn_dailymail", "3.0.0", "train", "article")
]

# 定义清理显存的函数
def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

# 定义处理单个模型的函数
def process_single_model(model_name):
    """处理单个模型的所有数据集测试"""
    print(f"\n{'='*80}")
    print(f"开始处理模型: {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 记录开始时间
    model_start_time = time.time()
    
    # 加载模型和tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 使用半精度以节省显存
            device_map="auto"  # 自动设备映射
        )
        model.eval()
        print("模型加载成功！")
        print(f"模型类型: {type(model).__name__}")
        clear_gpu_memory()
        
    except Exception as e:
        print(f"加载模型失败：{e}")
        clear_gpu_memory()
        return False
    
    # 定义钩子函数
    activation_percentages_per_layer = []
    handles = []
    
    def hook_fn(layer_idx):
        def fn(module, input, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                
                # 计算激活百分比
                if act.dim() == 3:
                    activated = (act.abs() > activation_threshold).float()
                    activation_rate = activated.mean(dim=[0, 1])
                    activation_pct = activation_rate.mean().item() * 100
                else:
                    activated = (act.abs() > activation_threshold).float()
                    activation_pct = activated.mean().item() * 100
                
                activation_percentages_per_layer[layer_idx].append(activation_pct)
                
                # 清理内存
                del act, activated
                if 'activation_rate' in locals():
                    del activation_rate
        return fn
    
    # 注册钩子
    try:
        model_type = type(model).__name__.lower()
        print(f"检测到的模型类型: {model_type}")
        
        # 根据模型类型选择正确的层路径
        if "opt" in model_type:
            num_layers = len(model.model.decoder.layers)
            for i in range(num_layers):
                hook = model.model.decoder.layers[i].fc2.register_forward_hook(hook_fn(i))
                handles.append(hook)
        elif "chatglm" in model_type:
            num_layers = len(model.transformer.encoder.layers)
            for i in range(num_layers):
                hook = model.transformer.encoder.layers[i].mlp.register_forward_hook(hook_fn(i))
                handles.append(hook)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            for i in range(num_layers):
                layer = model.model.layers[i]
                if hasattr(layer, 'mlp'):
                    hook = layer.mlp.register_forward_hook(hook_fn(i))
                    handles.append(hook)
                elif hasattr(layer, 'feed_forward'):
                    hook = layer.feed_forward.register_forward_hook(hook_fn(i))
                    handles.append(hook)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
            for i in range(num_layers):
                hook = model.transformer.h[i].mlp.register_forward_hook(hook_fn(i))
                handles.append(hook)
        else:
            print(f"无法自动检测模型 {model_name} 的层结构")
            return False
            
        print(f"模型总层数: {num_layers}")
        print(f"成功注册 {len(handles)} 个钩子")
        
    except Exception as e:
        print(f"注册钩子失败: {e}")
        # 清理已加载的模型
        del model
        clear_gpu_memory()
        return False
    
    # 存储所有数据集的结果
    all_dataset_results = {}
    
    # 循环每个数据集
    for ds_name, ds_config, ds_split, text_column in datasets_to_use:
        print(f"\n--- 处理数据集: {ds_name} (split: {ds_split}) ---")
        dataset_start_time = time.time()
        
        try:
            # 加载数据集
            if ds_config:
                dataset = load_dataset(ds_name, ds_config, split=ds_split)
            else:
                dataset = load_dataset(ds_name, split=ds_split)
            
            # 提取样本文本
            text_list = dataset[text_column][:num_samples]
            
            # 特殊处理squad数据集
            if ds_name == "squad":
                context_list = dataset['context'][:num_samples]
                text_list = [q + " " + c for q, c in zip(text_list, context_list)]
            
            samples = [text for text in text_list if isinstance(text, str) and text.strip()]
            if len(samples) == 0:
                print("跳过: 无有效样本")
                continue
            print(f"使用 {len(samples)} 个样本进行分析")
            
            # 重置激活记录
            activation_percentages_per_layer = [[] for _ in range(num_layers)]
            
            # 批量运行推理
            with torch.no_grad():
                for i, text in enumerate(tqdm(samples, desc="处理样本")):
                    if not text.strip():
                        continue
                    text = text[:max_length]
                    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                                     truncation=True, padding=True).to(device)
                    try:
                        outputs = model(**inputs)
                        del outputs
                    except Exception as e:
                        print(f"推理错误: {e}")
                        continue
                    finally:
                        del inputs
                    
                    # 定期清理显存
                    if (i + 1) % batch_size == 0:
                        clear_gpu_memory()
            
            # 清理显存
            clear_gpu_memory()
            
            # 计算统计数据
            layer_avg_activations = []
            for layer_idx in range(num_layers):
                layer_percentages = activation_percentages_per_layer[layer_idx]
                if layer_percentages:
                    avg_pct = np.mean(layer_percentages)
                    layer_avg_activations.append({
                        "layer_id": layer_idx,
                        "avg_activation_percentage": round(avg_pct, 4),
                        "num_samples": len(layer_percentages)
                    })
                else:
                    layer_avg_activations.append({
                        "layer_id": layer_idx,
                        "avg_activation_percentage": 0.0,
                        "num_samples": 0
                    })
            
            # 计算整体平均
            all_layer_percentages = [
                layer_data["avg_activation_percentage"] 
                for layer_data in layer_avg_activations 
                if layer_data["num_samples"] > 0
            ]
            overall_avg = np.mean(all_layer_percentages) if all_layer_percentages else 0.0
            
            # 保存结果
            dataset_result = {
                "dataset": ds_name,
                "split": ds_split,
                "num_samples": len(samples),
                "activation_threshold": activation_threshold,
                "overall_avg_activation_percentage": round(overall_avg, 4),
                "layers": layer_avg_activations,
                "processing_time": round(time.time() - dataset_start_time, 2)
            }
            
            all_dataset_results[ds_name] = dataset_result
            
            # 打印摘要
            print(f"\n{ds_name} 数据集结果:")
            print(f"  - 整体平均激活百分比: {overall_avg:.2f}%")
            print(f"  - 处理时间: {dataset_result['processing_time']} 秒")
            
            # 清理变量
            del dataset, text_list, samples
            if ds_name == "squad" and 'context_list' in locals():
                del context_list
            clear_gpu_memory()
            
        except Exception as e:
            print(f"处理 {ds_name} 失败: {e}")
            clear_gpu_memory()
    
    # 保存结果
    model_folder = model_name.replace("/", "_").replace("-", "_")
    os.makedirs(model_folder, exist_ok=True)
    
    # 详细结果
    summary_filename = "activation_summary_all_datasets.json"
    summary_path = os.path.join(model_folder, summary_filename)
    summary_results = {
        "model": model_name,
        "model_type": model_type,
        "num_layers": num_layers,
        "activation_threshold": activation_threshold,
        "total_processing_time": round(time.time() - model_start_time, 2),
        "datasets": all_dataset_results
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\n综合结果保存为: {summary_path}")
    
    # 简化比较结果
    comparison_filename = "activation_comparison.json"
    comparison_path = os.path.join(model_folder, comparison_filename)
    comparison_data = {
        "model": model_name,
        "activation_threshold": activation_threshold,
        "dataset_averages": {}
    }
    for ds_name, result in all_dataset_results.items():
        comparison_data["dataset_averages"][ds_name] = {
            "overall_activation_percentage": result["overall_avg_activation_percentage"],
            "num_samples": result["num_samples"]
        }
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    # 删除模型并清理显存
    del model
    if 'tokenizer' in locals():
        del tokenizer
    clear_gpu_memory()
    
    print(f"\n模型 {model_name} 处理完成！")
    print(f"总处理时间: {round(time.time() - model_start_time, 2)} 秒")
    
    return True

# 主程序：循环处理所有模型
def main():
    """主函数：循环处理所有模型"""
    print("="*80)
    print("批量模型激活稀疏度测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待测试模型数量: {len(MODEL_LIST)}")
    print("="*80)
    
    # 创建总结果文件夹
    os.makedirs("all_models_results", exist_ok=True)
    
    # 记录处理状态
    processing_log = {
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "models": {}
    }
    
    successful_models = []
    failed_models = []
    
    # 循环处理每个模型
    for idx, model_name in enumerate(MODEL_LIST, 1):
        print(f"\n\n{'#'*80}")
        print(f"进度: {idx}/{len(MODEL_LIST)}")
        print(f"{'#'*80}")
        
        try:
            success = process_single_model(model_name)
            if success:
                successful_models.append(model_name)
                processing_log["models"][model_name] = "成功"
            else:
                failed_models.append(model_name)
                processing_log["models"][model_name] = "失败"
        except Exception as e:
            print(f"处理模型 {model_name} 时发生严重错误: {e}")
            failed_models.append(model_name)
            processing_log["models"][model_name] = f"错误: {str(e)}"
        
        # 每个模型处理完后都清理显存
        clear_gpu_memory()
        
        # 短暂等待，确保资源完全释放
        time.sleep(5)
    
    # 保存处理日志
    processing_log["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    processing_log["successful_models"] = successful_models
    processing_log["failed_models"] = failed_models
    processing_log["success_rate"] = f"{len(successful_models)}/{len(MODEL_LIST)}"
    
    log_path = os.path.join("all_models_results", "processing_log.json")
    with open(log_path, 'w') as f:
        json.dump(processing_log, f, indent=4)
    
    # 打印最终统计
    print("\n\n" + "="*80)
    print("所有模型处理完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功: {len(successful_models)} 个模型")
    print(f"失败: {len(failed_models)} 个模型")
    
    if successful_models:
        print("\n成功的模型:")
        for model in successful_models:
            print(f"  ✓ {model}")
    
    if failed_models:
        print("\n失败的模型:")
        for model in failed_models:
            print(f"  ✗ {model}")
    
    print(f"\n处理日志保存在: {log_path}")
    print("="*80)

# 运行主程序
if __name__ == "__main__":
    main()