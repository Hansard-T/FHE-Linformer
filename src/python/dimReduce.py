import os
import argparse
from socket import EAI_BADFLAGS
import numpy as np
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_numeric_txt(path: str, array: np.ndarray):
    # 保存为纯数字（逗号分隔），二维或一维数组逐行保存
    ensure_dir(os.path.dirname(path))
    np.savetxt(path, array.reshape(array.shape[0], -1), delimiter=',', fmt='%.18e')


def list_input_files(dir_path: str):
    files = []
    for name in os.listdir(dir_path):
        if name.startswith('input_') and name.endswith('.txt'):
            try:
                idx = int(name[len('input_'):-len('.txt')])
                files.append((idx, os.path.join(dir_path, name)))
            except ValueError:
                continue
    files.sort(key=lambda x: x[0])
    return files


def load_input_vectors(dir_path: str, files_sorted: list) -> np.ndarray:
    vecs = []
    for idx, fp in files_sorted:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"找不到输入文件: {fp}")
        v = np.loadtxt(fp, dtype=np.float32)
        vecs.append(v.reshape(1, -1))
    return np.vstack(vecs)


def main():
    parser = argparse.ArgumentParser(description='使用Linformer(20NG)权重对单条样本执行前向并保存数值文件')
    parser.add_argument('--tokens_dir', default='/Users/tangxianning/Downloads/FHE-Linformer/src/tmp_embeddings', help='包含tokens.txt的目录')
    parser.add_argument('--ckpt_path', default='/Users/tangxianning/Downloads/FHE-Linformer/src/trained_models/Linformer_L1H128A2_p32_for_20NG_cv1_906.pkl', help='Linformer检查点路径')
    parser.add_argument('--precompress', action='store_true', help='先用E/F在序列维度对输入降维（与先算K/V后乘E/F数值等价）')
    parser.add_argument('--use_inputs', action='store_true', help='从 tokens_dir 中的 input_*.txt 直接读取每个位置的输入向量')
    args = parser.parse_args()

    input_files = list_input_files(args.tokens_dir)
    if not input_files:
        raise FileNotFoundError(f"在 {args.tokens_dir} 没有找到 input_*.txt 文件")

    S = len(input_files)
    x_emb = load_input_vectors(args.tokens_dir, input_files)  # [S, D]

    # 加载检查点权重
    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model']

    # 取需要的权重为numpy
    posEmb = sd['posEmb'].detach().cpu().numpy()  # [700, 128]

    # 低秩投影矩阵 E/F（按S切片列），使用二维数组
    E_w = sd['linformer.transformerLayers.transformer0.selfAttn.E.weight'].detach().cpu().numpy()
    F_w = sd['linformer.transformerLayers.transformer0.selfAttn.F.weight'].detach().cpu().numpy()

    E_b = sd['linformer.transformerLayers.transformer0.selfAttn.E.bias'].detach().cpu().numpy()
    F_b = sd['linformer.transformerLayers.transformer0.selfAttn.F.bias'].detach().cpu().numpy()

    cls_token = sd['cls_token'].detach().cpu().numpy().reshape(1, 128)

    pos_slice = posEmb[:S] / 3.0
    x_main_in = x_emb + pos_slice
    x_in = np.vstack([cls_token, x_main_in])
    S_total = x_in.shape[0]

    Eh = E_w[:, :S_total]  # [32, S_total]
    Fh = F_w[:, :S_total]  # [32, S_total]
    # 广播加偏置到每个特征维度
    X_E = (Eh @ x_in) + E_b[:, None]
    X_F = (Fh @ x_in) + F_b[:, None]
    # 按压缩后序列维度 p 输出（例如32）
    p = Eh.shape[0]
    for i in range(p):
        save_numeric_txt(os.path.join('/Users/tangxianning/Downloads/FHE-Linformer/input', f'XE_{i}.txt'), X_E[i])
        save_numeric_txt(os.path.join('/Users/tangxianning/Downloads/FHE-Linformer/input', f'XF_{i}.txt'), X_F[i])

if __name__ == '__main__':
    main()