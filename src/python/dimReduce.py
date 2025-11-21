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


def load_tokenizer_for_20ng(data_root: str, seq_max_len: int = 700):
    # 延迟导入，避免 --use_inputs 模式下引入 mittens 依赖
    from utils import Sklearn_20NG
    from tokenizerFuncs import Tokenizer
    trainDS = Sklearn_20NG(data_root, 'train')
    testDS = Sklearn_20NG(data_root, 'test')
    tokenizer = Tokenizer(trainDS.sequences + testDS.sequences,
                          trainDS.labels + testDS.labels,
                          seqMaxLen=seq_max_len)
    return tokenizer


def read_tokens(tokens_dir: str):
    tokens_path = os.path.join(tokens_dir, 'tokens.txt')
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(f"找不到 tokens.txt: {tokens_path}")
    with open(tokens_path, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens


def gelu(x: np.ndarray) -> np.ndarray:
    # 使用近似 GELU
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # x: [S, D], gamma/beta: [D]
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * gamma + beta


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


def load_txt(path, expect_cols=None):
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception:
        arr = np.loadtxt(path)
    if arr.ndim == 1:
        if expect_cols is not None:
            total = arr.size
            if total % expect_cols == 0:
                arr = arr.reshape(total // expect_cols, expect_cols)
            else:
                return arr.astype(np.float32)
        else:
            return arr.astype(np.float32)
    if expect_cols is not None and arr.shape[1] != expect_cols:
        raise ValueError(f"文件 {path} 列数 {arr.shape[1]} != 期望 {expect_cols}")
    return arr.astype(np.float32)


def load_matrix(path, expect_shape):
    rows, cols = expect_shape
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception:
        arr = np.loadtxt(path)
    if arr.ndim == 1:
        total = arr.size
        if total % rows == 0:
            c2 = total // rows
            arr = arr.reshape(rows, c2)
            if c2 > cols:
                arr = arr[:, :cols]
            elif c2 < cols:
                raise ValueError(f"文件 {path} 列数 {c2} 少于期望 {cols}")
        else:
            raise ValueError(f"文件 {path} 元素数 {total} 无法按 {rows} 行整除")
    elif arr.shape[0] == rows and arr.shape[1] != cols:
        if arr.shape[1] > cols:
            arr = arr[:, :cols]
        else:
            raise ValueError(f"文件 {path} 列数 {arr.shape[1]} 少于期望 {cols}")
    elif arr.shape != (rows, cols):
        raise ValueError(f"文件 {path} 形状 {arr.shape} 不等于期望 {(rows, cols)}")
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='使用 tmp_embeddings/input_*.txt 与 weights-20NG 文本权重进行前向推理')
    parser.add_argument('--tokens_dir', default='/Users/tangxianning/Downloads/FHE-Linformer/src/tmp_embeddings', help='包含 input_*.txt 的目录')
    parser.add_argument('--weights_dir', default='/Users/tangxianning/Downloads/FHE-Linformer/weights-20NG', help='文本权重目录')
    args = parser.parse_args()

    files_sorted = list_input_files(args.tokens_dir)
    if not files_sorted:
        raise FileNotFoundError(f"在 {args.tokens_dir} 没有找到 input_*.txt 文件")
    x_emb = load_input_vectors(args.tokens_dir, files_sorted)  # 形状: [S,128]
    S = x_emb.shape[0]  # 形状: 标量

    posEmb = load_matrix(os.path.join(args.weights_dir, 'posEmb.txt'), (700, 128))  # 形状: [700,128]
    cls_token = load_txt(os.path.join(args.weights_dir, 'cls_token.txt'))  # 形状: [128]

    s = 1
    pos_slice = posEmb[:S] / 3.0  # 形状: [S,128]
    x_main_in = x_emb + pos_slice  # 形状: [S,128]
    x_in = np.vstack([cls_token.reshape(1, -1), x_main_in])  # 形状: [S+1,128]
    x_in = s * x_in  # 形状: [S+1,128]
    S_total = x_in.shape[0]  # 形状: 标量

    E_w = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_E_weight.txt'), (32, 701))  # 形状: [32,701]
    E_b = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_E_bias.txt'))    # 形状: [32]
    F_w = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_F_weight.txt'), (32, 701))  # 形状: [32,701]
    F_b = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_F_bias.txt'))    # 形状: [32]

    Eh = E_w[:, :S_total]  # 形状: [32,S_total]
    Fh = F_w[:, :S_total]  # 形状: [32,S_total]
    X_E = Eh @ x_in + E_b.reshape(-1, 1) * s   # 形状: [32,128]
    X_F = Fh @ x_in + F_b.reshape(-1, 1) * s   # 形状: [32,128]

    for i in range(X_E.shape[0]):
        save_numeric_txt(os.path.join('/Users/tangxianning/Downloads/FHE-Linformer/input', f'XE_{i}.txt'), X_E[i])
        save_numeric_txt(os.path.join('/Users/tangxianning/Downloads/FHE-Linformer/input', f'XF_{i}.txt'), X_F[i])

if __name__ == '__main__':
    main()