import os
import argparse
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
    parser.add_argument('--ckpt_path', default='/Users/tangxianning/Downloads/FHE-Linformer/src/trained_models/Linformer_L1H512A2_p32_for_20NG_cv1_903.pkl', help='Linformer检查点路径')
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

    # 低秩投影矩阵 E/F（按S切片列）
    E = [sd['linformer.seq.0.fn.heads.0.E.weight'].detach().cpu().numpy(),
         sd['linformer.seq.0.fn.heads.1.E.weight'].detach().cpu().numpy()]  # [32,700]
    F = [sd['linformer.seq.0.fn.heads.0.F.weight'].detach().cpu().numpy(),
         sd['linformer.seq.0.fn.heads.1.F.weight'].detach().cpu().numpy()]  # [32,700]

    pos_slice = posEmb[:S] / 3.0
    x_in = x_emb + pos_slice

    for h in range(2):
        Eh = E[h][:, :S]  # [32,S]
        Fh = F[h][:, :S]  # [32,S]
        X_E = Eh @ x_in           # [32,128]
        X_F = Fh @ x_in           # [32,128]
        save_numeric_txt(os.path.join('/Users/tangxianning/Downloads/FHE-Linformer/input', f'XE_{h}.txt'), X_E)
        save_numeric_txt(os.path.join('/Users/tangxianning/Downloads/FHE-Linformer/input', f'XF_{h}.txt'), X_F)

if __name__ == '__main__':
    main()