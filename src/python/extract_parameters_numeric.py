import torch
import os
import numpy as np

def extract_model_parameters_numeric():
    ckpt_path = "../../src/trained_models/Linformer_L1H128A2_p32_for_20NG_cv1_845.pkl"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_state = ckpt['model']

    output_dir = "weights-20NG"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for param_name, param_tensor in model_state.items():
        safe_param_name = param_name.replace('.', '_').replace('/', '_')
        txt_filepath = os.path.join(output_dir, f"{safe_param_name}.txt")

        arr = param_tensor.detach().cpu().numpy()
        if np.isscalar(arr) or getattr(arr, 'ndim', 0) == 0:
            arr2 = np.array([[float(arr)]], dtype=np.float32)
        elif arr.ndim == 1:
            arr2 = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            arr2 = arr
        else:
            arr2 = arr.reshape(arr.shape[0], -1)

        np.savetxt(txt_filepath, arr2, fmt='%.18e', delimiter=',')
        print(f"已保存参数 '{param_name}' 到文件: {txt_filepath} (形状: {param_tensor.shape}, 元素数: {param_tensor.numel()})")

    print(f"\n所有参数已保存到目录: {output_dir}")
    print(f"共保存了 {len(model_state)} 个参数文件")

if __name__ == "__main__":
    extract_model_parameters_numeric()