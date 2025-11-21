from math import exp
import os
import argparse
import numpy as np
import torch



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
    x_in = x_in * s
    S_total = x_in.shape[0]  # 形状: 标量

    E_w = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_E_weight.txt'), (32, 701))  # 形状: [32,701]
    E_b = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_E_bias.txt'))    # 形状: [32]
    F_w = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_F_weight.txt'), (32, 701))  # 形状: [32,701]
    F_b = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_F_bias.txt'))    # 形状: [32]

    Eh = E_w[:, :S_total]  # 形状: [32,S_total]
    Fh = F_w[:, :S_total]  # 形状: [32,S_total]
    X_E = Eh @ x_in + E_b.reshape(-1, 1) * s  # 形状: [32,128]
    X_F = Fh @ x_in + F_b.reshape(-1, 1) * s  # 形状: [32,128]

    WQ = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WQ_weight.txt'), (128, 128))  # 形状: [128,128]
    BQ = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WQ_bias.txt'))    # 形状: [128]
    WK = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WK_weight.txt'), (128, 128))  # 形状: [128,128]
    BK = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WK_bias.txt'))    # 形状: [128]
    WV = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WV_weight.txt'), (128, 128))  # 形状: [128,128]
    BV = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WV_bias.txt'))    # 形状: [128]

    Q = x_in @ WQ.T + BQ.reshape(1, -1) * s  # 形状: [S_total,128]

    K = X_E @ WK.T + BK.reshape(1, -1) * s         # 形状: [32,128]

    V = X_F @ WV.T + BV.reshape(1, -1) * s         # 形状: [32,128]

    logits = Q @ K.T            # 形状: [S_total,32]
    print("logits:", logits[1])
    r = 1.0 / 8.0
    x = logits * r              # 形状: [S_total,32]
    exp_approx = (
        1 + x + (x**2) * 0.5 + (x**3) * (1/6.0) + (x**4) * (1/24.0) + (x**5) * (1/120.0) + (x**6) * (1/720.0)
    )                           # 形状: [S_total,32]

    attn = exp_approx / exp_approx.sum(axis=-1, keepdims=True)  # 形状: [S_total,32]
    print("approx_max:", exp_approx.sum(axis=-1))
    print("approx_min:", exp_approx.sum(axis=-1))
    O = attn @ V                                # 形状: [S_total,128]

    WO = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WO_weight.txt'), (128, 128))  # 形状: [128,128]
    BO = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_selfAttn_WO_bias.txt'))    # 形状: [128]
    attn_out = O @ WO.T + BO.reshape(1, -1) * s * s    # 形状: [S_total,128]

    x_attn_res = x_in + attn_out  # 形状: [S_total,128]

    c10 = float(np.loadtxt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine1_c0.txt')))  # 形状: 标量
    c11 = float(np.loadtxt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine1_c1.txt')))  # 形状: 标量
    c12 = float(np.loadtxt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine1_c2.txt')))  # 形状: 标量
    fL1 = c10 + c11 / np.sqrt(S_total) + c12 / S_total  # 形状: 标量
    a1 = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine1_a.txt')) * fL1  # 形状: [128]
    b1 = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine1_b.txt')) * fL1  # 形状: [128]
    x_norm0 = x_attn_res * a1.reshape(1, -1) + b1.reshape(1, -1) * s * s # 形状: [S_total,128]

    Wffn0 = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_Wffn_0_weight.txt'), (512, 128))  # 形状: [512,128]
    Bffn0 = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_Wffn_0_bias.txt'))    # 形状: [512]
    Wffn2 = load_matrix(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_Wffn_2_weight.txt'), (128, 512))  # 形状: [128,512]
    Bffn2 = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_Wffn_2_bias.txt'))    # 形状: [128]
    ff_hidden = gelu((x_norm0 / s) @ Wffn0.T + Bffn0.reshape(1, -1)) * s * s # 形状: [S_total,512]

    ff_out = ff_hidden @ Wffn2.T + Bffn2.reshape(1, -1) * s * s # 形状: [S_total,128]

    x_ff_res = x_norm0 + ff_out  # 形状: [S_total,128]

    c20 = float(np.loadtxt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine2_c0.txt')))  # 形状: 标量
    c21 = float(np.loadtxt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine2_c1.txt')))  # 形状: 标量
    c22 = float(np.loadtxt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine2_c2.txt')))  # 形状: 标量
    fL2 = c20 + c21 / np.sqrt(S_total) + c22 / S_total  # 形状: 标量
    a2 = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine2_a.txt')) * fL2  # 形状: [128]
    b2 = load_txt(os.path.join(args.weights_dir, 'linformer_transformerLayers_transformer0_ffn_affine2_b.txt')) * fL2  # 形状: [128]
    x_norm1 = x_ff_res * a2.reshape(1, -1) + b2.reshape(1, -1) * s * s # 形状: [S_total,128]
   
    Wp = load_matrix(os.path.join(args.weights_dir, 'pooler_dense_weight.txt'), (128, 128))  # 形状: [128,128]
    bp = load_txt(os.path.join(args.weights_dir, 'pooler_dense_bias.txt'))  # 形状: [128]
    cls = np.tanh(x_norm1 @ Wp.T + bp.reshape(1, -1)) # 形状: [1,128]

    fc_w = load_matrix(os.path.join(args.weights_dir, 'fcLinear_0_weight.txt'), (20, 128))  # 形状: [20,128]
    fc_b = load_txt(os.path.join(args.weights_dir, 'fcLinear_0_bias.txt'))  # 形状: [20]
    y_logit = cls @ fc_w.T + fc_b.reshape(1, -1) * s * s  # 形状: [1,20]


    logits_max = (y_logit / s * s).max(axis=-1, keepdims=True)  # 形状: [1,1]
    y_prob = np.exp((y_logit / s * s) - logits_max)  # 形状: [1,20]
    y_prob = y_prob / y_prob.sum(axis=-1, keepdims=True)  # 形状: [1,20]
    print(y_prob)
    y_pred = int(np.argmax(y_prob, axis=-1)[0])  # 形状: 标量

    print('Forward done. Shapes:')
    print('x_in', x_in.shape, 'X_E', X_E.shape, 'X_F', X_F.shape)
    print('Q', Q.shape, 'K', K.shape, 'V', V.shape, 'attn_out', attn_out.shape)
    print('x_norm1', x_norm1.shape, 'y_logit', y_logit.shape)
    print('Pred:', y_pred, 'Prob:', float(y_prob[0, y_pred]))

    try:
        tokenizer = load_tokenizer_for_20ng('/Users/tangxianning/Downloads/FHE-Linformer/datasets/20NG')
        id2lab = tokenizer.id2lab
        if 0 <= y_pred < len(id2lab):
            print('Pred Label:', id2lab[y_pred])
    except Exception as e:
        pass


if __name__ == '__main__':
    main()
