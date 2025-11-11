import torch
import os
import numpy as np

def extract_model_parameters_numeric():
    """提取Linformer模型参数并保存为纯数字格式的txt文件"""
    
    # 加载模型
    ckpt_path = "../trained_models/Linformer_L1H128A2_p32_for_20NG_cv1_906.pkl"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    model_state = ckpt['model']
    print(f"模型参数类型: {type(model_state)}")
    print(f"总参数数量: {len(model_state)}")
    
    # 创建输出目录
    output_dir = "../../weights-20NG"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历所有参数并保存为纯数字格式
    for param_name, param_tensor in model_state.items():
        # 将参数名中的特殊字符替换为下划线，以便作为文件名
        safe_param_name = param_name.replace('.', '_').replace('/', '_')
        
        # 创建txt文件名
        txt_filename = f"{safe_param_name}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        # 将tensor转换为numpy数组
        param_array = param_tensor.detach().cpu().numpy()
        
        # 保存为纯数字格式
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            # 将数组展平并保存为逗号分隔的数字
            flat_array = param_array.flatten()
            
            # 使用科学计数法格式，保持高精度
            numeric_strings = []
            for val in flat_array:
                # 使用%.18e格式保持最高精度
                numeric_strings.append(f"{val:.18e}")
            
            # 将所有数字用逗号连接成一行
            f.write(','.join(numeric_strings))
        
        print(f"已保存参数 '{param_name}' 到文件: {txt_filepath} (形状: {param_tensor.shape}, 元素数: {param_tensor.numel()})")
    
    print(f"\n所有参数已保存到目录: {output_dir}")
    print(f"共保存了 {len(model_state)} 个参数文件")
    print("每个文件包含该参数的所有数值，以逗号分隔，使用科学计数法格式")

if __name__ == "__main__":
    extract_model_parameters_numeric()