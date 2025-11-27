from math import exp
import os
import argparse
import numpy as np
import torch
from utils import Sklearn_20NG



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_tokenizer_for_20ng(data_root: str, seq_max_len: int = 700):
    # 延迟导入，避免 --use_inputs 模式下引入 mittens 依赖
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
    parser = argparse.ArgumentParser(description='使用 tmp_embeddings/input_*.txt 与 weights-20NG 文本权重进行前向推理或批量评估')
    parser.add_argument('--tokens_dir', default='/Users/tangxianning/Downloads/FHE-Linformer/src/tmp_embeddings')
    parser.add_argument('--weights_dir', default='/Users/tangxianning/Downloads/FHE-Linformer/weights-20NG')
    parser.add_argument('--eval_root', default='')
    parser.add_argument('--max_eval', type=int, default=0)
    parser.add_argument('--cls_only', action='store_true')
    args = parser.parse_args()

    def load_all_weights(wd):
        posEmb = load_matrix(os.path.join(wd, 'posEmb.txt'), (700, 128))
        cls_token = load_txt(os.path.join(wd, 'cls_token.txt'))
        E_w = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_E_weight.txt'), (32, 701))
        E_b = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_E_bias.txt'))
        F_w = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_F_weight.txt'), (32, 701))
        F_b = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_F_bias.txt'))
        WQ = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WQ_weight.txt'), (128, 128))
        BQ = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WQ_bias.txt'))
        WK = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WK_weight.txt'), (128, 128))
        BK = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WK_bias.txt'))
        WV = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WV_weight.txt'), (128, 128))
        BV = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WV_bias.txt'))
        WO = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WO_weight.txt'), (128, 128))
        BO = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_selfAttn_WO_bias.txt'))
        c10 = float(np.loadtxt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine1_c0.txt')))
        c11 = float(np.loadtxt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine1_c1.txt')))
        c12 = float(np.loadtxt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine1_c2.txt')))
        a1 = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine1_a.txt'))
        b1 = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine1_b.txt'))
        Wffn0 = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_Wffn_0_weight.txt'), (512, 128))
        Bffn0 = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_Wffn_0_bias.txt'))
        Wffn2 = load_matrix(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_Wffn_2_weight.txt'), (128, 512))
        Bffn2 = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_Wffn_2_bias.txt'))
        c20 = float(np.loadtxt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine2_c0.txt')))
        c21 = float(np.loadtxt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine2_c1.txt')))
        c22 = float(np.loadtxt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine2_c2.txt')))
        a2 = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine2_a.txt'))
        b2 = load_txt(os.path.join(wd, 'linformer_transformerLayers_transformer0_ffn_affine2_b.txt'))
        Wp = load_matrix(os.path.join(wd, 'pooler_dense_weight.txt'), (128, 128))
        bp = load_txt(os.path.join(wd, 'pooler_dense_bias.txt'))
        fc_w = load_matrix(os.path.join(wd, 'fcLinear_0_weight.txt'), (20, 128))
        fc_b = load_txt(os.path.join(wd, 'fcLinear_0_bias.txt'))
        return {
            'posEmb': posEmb, 'cls_token': cls_token, 'E_w': E_w, 'E_b': E_b, 'F_w': F_w, 'F_b': F_b,
            'WQ': WQ, 'BQ': BQ, 'WK': WK, 'BK': BK, 'WV': WV, 'BV': BV, 'WO': WO, 'BO': BO,
            'c10': c10, 'c11': c11, 'c12': c12, 'a1': a1, 'b1': b1,
            'Wffn0': Wffn0, 'Bffn0': Bffn0, 'Wffn2': Wffn2, 'Bffn2': Bffn2,
            'c20': c20, 'c21': c21, 'c22': c22, 'a2': a2, 'b2': b2,
            'Wp': Wp, 'bp': bp, 'fc_w': fc_w, 'fc_b': fc_b
        }

    def forward_tokens_dir(tokens_dir, weights, cls_only):
        files_sorted = list_input_files(tokens_dir)  # 形状: 列表[(idx, 路径)]
        if not files_sorted:
            return None
        x_emb = load_input_vectors(tokens_dir, files_sorted)  # 形状: [S,128]
        S = x_emb.shape[0]  # 形状: 标量S
        s = 1.0  # 形状: 标量
        pos_slice = weights['posEmb'][:S] / 3.0  # 形状: [S,128]
        x_main_in = x_emb + pos_slice  # 形状: [S,128]
        x_in = np.vstack([weights['cls_token'].reshape(1, -1), x_main_in])  # 形状: [S+1,128]
        x_in = x_in * s  # 形状: [S+1,128]
        S_total = x_in.shape[0]  # 形状: 标量(S+1)
        Eh = weights['E_w'][:, :S_total]  # 形状: [32,S_total]
        Fh = weights['F_w'][:, :S_total]  # 形状: [32,S_total]
        X_E = Eh @ x_in + weights['E_b'].reshape(-1, 1) * s  # 形状: [32,128]
        X_F = Fh @ x_in + weights['F_b'].reshape(-1, 1) * s  # 形状: [32,128]
        Q = x_in @ weights['WQ'].T + weights['BQ'].reshape(1, -1) * s  # 形状: [S_total,128]
        K = X_E @ weights['WK'].T + weights['BK'].reshape(1, -1) * s  # 形状: [32,128]
        V = X_F @ weights['WV'].T + weights['BV'].reshape(1, -1) * s  # 形状: [32,128]
        r = 1.0 / 8.0  # 形状: 标量
        if cls_only:
            logits_cls = Q[0:1, :] @ K.T  # 形状: [1,32]
            x_cls = logits_cls * r  # 形状: [1,32]
            exp_approx_cls = 1 + x_cls + (x_cls**2) * 0.5 + (x_cls**3) * (1/6.0) + (x_cls**4) * (1/24.0) + (x_cls**5) * (1/120.0) + (x_cls**6) * (1/720.0)  # 形状: [1,32]
            attn_cls = exp_approx_cls / exp_approx_cls.sum(axis=-1, keepdims=True)  # 形状: [1,32]
            print(exp_approx_cls.sum(axis=-1))
            O_cls = attn_cls @ V  # 形状: [1,128]
            attn_out_cls = O_cls @ weights['WO'].T + weights['BO'].reshape(1, -1) * s * s  # 形状: [1,128]
            attn_out = np.zeros_like(x_in)  # 形状: [S_total,128]
            attn_out[0:1, :] = attn_out_cls  # 形状: [1,128]
        else:
            logits = Q @ K.T  # 形状: [S_total,32]
            x = logits * r  # 形状: [S_total,32]
            exp_approx = 1 + x + (x**2) * 0.5 + (x**3) * (1/6.0) + (x**4) * (1/24.0) + (x**5) * (1/120.0) + (x**6) * (1/720.0)  # 形状: [S_total,32]
            attn = exp_approx / exp_approx.sum(axis=-1, keepdims=True)  # 形状: [S_total,32]
            O = attn @ V  # 形状: [S_total,128]
            attn_out = O @ weights['WO'].T + weights['BO'].reshape(1, -1) * s * s  # 形状: [S_total,128]
        x_attn_res = x_in + attn_out  # 形状: [S_total,128]
        fL1 = weights['c10'] + weights['c11'] / np.sqrt(S_total) + weights['c12'] / S_total  # 形状: 标量
        a1 = weights['a1'] * fL1  # 形状: [128]
        b1 = weights['b1'] * fL1  # 形状: [128]
        x_norm0 = x_attn_res * a1.reshape(1, -1) + b1.reshape(1, -1) * s * s  # 形状: [S_total,128]
        ff_hidden = gelu((x_norm0 / s) @ weights['Wffn0'].T + weights['Bffn0'].reshape(1, -1)) * s * s  # 形状: [S_total,512]
        ff_out = ff_hidden @ weights['Wffn2'].T + weights['Bffn2'].reshape(1, -1) * s * s  # 形状: [S_total,128]
        x_ff_res = x_norm0 + ff_out  # 形状: [S_total,128]
        fL2 = weights['c20'] + weights['c21'] / np.sqrt(S_total) + weights['c22'] / S_total  # 形状: 标量
        a2 = weights['a2'] * fL2  # 形状: [128]
        b2 = weights['b2'] * fL2  # 形状: [128]
        x_norm1 = x_ff_res * a2.reshape(1, -1) + b2.reshape(1, -1) * s * s  # 形状: [S_total,128]
        cls = np.tanh(x_norm1[0:1, :] @ weights['Wp'].T + weights['bp'].reshape(1, -1))  # 形状: [1,128]
        y_logit = cls @ weights['fc_w'].T + weights['fc_b'].reshape(1, -1) * s * s  # 形状: [1,20]
        logits_max = (y_logit / s * s).max(axis=-1, keepdims=True)  # 形状: [1,1]
        y_prob = np.exp((y_logit / s * s) - logits_max)  # 形状: [1,20]
        y_prob = y_prob / y_prob.sum(axis=-1, keepdims=True)  # 形状: [1,20]
        y_pred = int(np.argmax(y_prob, axis=-1)[0])  # 形状: 标量
        return y_pred, float(y_prob[0, y_pred])  # 形状: (标量, 标量)

    if args.eval_root:
        weights = load_all_weights(args.weights_dir)  # 形状: 权重字典
        ds = Sklearn_20NG('/Users/tangxianning/Downloads/FHE-Linformer/datasets/20NG', 'test')  # 形状: 数据集对象
        tokenizer = load_tokenizer_for_20ng('/Users/tangxianning/Downloads/FHE-Linformer/datasets/20NG')  # 形状: 分词器对象
        id2lab = tokenizer.id2lab  # 形状: 列表[str]
        dirs = []  # 形状: 列表[(样本索引,int),(目录,str)]
        for name in os.listdir(args.eval_root):
            p = os.path.join(args.eval_root, name)
            if os.path.isdir(p) and name.startswith('test_'):
                try:
                    idx = int(name.split('_')[1])
                    dirs.append((idx, p))
                except Exception:
                    continue
        dirs.sort(key=lambda x: x[0])
        total = len(dirs)  # 形状: 标量
        if args.max_eval and args.max_eval > 0:
            total = min(total, args.max_eval)
            dirs = dirs[:total]
        correct = 0  # 形状: 标量
        for i, (idx, d) in enumerate(dirs):
            res = forward_tokens_dir(d, weights, args.cls_only)  # 形状: (pred,int),(prob,float)
            if res is None:
                continue
            y_pred, prob = res
            lab_true = ds.labels[idx]  # 形状: str
            lab_pred = id2lab[y_pred] if 0 <= y_pred < len(id2lab) else str(y_pred)  # 形状: str
            ok = (lab_true == lab_pred)  # 形状: 布尔
            if ok:
                correct += 1
            if i % 50 == 0:
                print(f"[{i}/{len(dirs)}] idx={idx} pred={lab_pred} prob={prob:.4f} true={lab_true}")
        acc = correct / len(dirs) if dirs else 0.0  # 形状: 标量
        print(f"Eval done. samples={len(dirs)} acc={acc:.6f}")
        return

    tokens_dir = args.tokens_dir
    files_sorted = list_input_files(tokens_dir)
    if not files_sorted:
        raise FileNotFoundError(f"在 {tokens_dir} 没有找到 input_*.txt 文件")
    weights = load_all_weights(args.weights_dir)
    y_pred, prob = forward_tokens_dir(tokens_dir, weights, args.cls_only)
    tokenizer = load_tokenizer_for_20ng('/Users/tangxianning/Downloads/FHE-Linformer/datasets/20NG')
    id2lab = tokenizer.id2lab
    print('Pred:', y_pred, 'Prob:', prob)
    if 0 <= y_pred < len(id2lab):
        print('Pred Label:', id2lab[y_pred])

if __name__ == '__main__':
    main()
