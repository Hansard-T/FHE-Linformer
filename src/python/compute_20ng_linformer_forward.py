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


def main():
    parser = argparse.ArgumentParser(description='使用Linformer(20NG)权重对单条样本执行前向并保存数值文件')
    parser.add_argument('--tokens_dir', default='/Users/tangxianning/Downloads/FHE-Linformer/src/tmp_embeddings', help='包含tokens.txt的目录')
    parser.add_argument('--data_root', default='/Users/tangxianning/Downloads/FHE-Linformer/datasets/20NG', help='20NG数据根目录')
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

    # 头部线性映射权重（每头一个Linear）
    to_q_w = [sd['linformer.seq.0.fn.to_q.0.weight'].detach().cpu().numpy(),
              sd['linformer.seq.0.fn.to_q.1.weight'].detach().cpu().numpy()]  # [256,128]
    to_k_w = [sd['linformer.seq.0.fn.to_k.0.weight'].detach().cpu().numpy(),
              sd['linformer.seq.0.fn.to_k.1.weight'].detach().cpu().numpy()]  # [256,128]
    to_v_w = [sd['linformer.seq.0.fn.to_v.0.weight'].detach().cpu().numpy(),
              sd['linformer.seq.0.fn.to_v.1.weight'].detach().cpu().numpy()]  # [256,128]

    # 低秩投影矩阵 E/F（按S切片列）
    E = [sd['linformer.seq.0.fn.heads.0.E.weight'].detach().cpu().numpy(),
         sd['linformer.seq.0.fn.heads.1.E.weight'].detach().cpu().numpy()]  # [32,700]
    F = [sd['linformer.seq.0.fn.heads.0.F.weight'].detach().cpu().numpy(),
         sd['linformer.seq.0.fn.heads.1.F.weight'].detach().cpu().numpy()]  # [32,700]

    # 输出投影与偏置
    w_o = sd['linformer.seq.0.fn.w_o.weight'].detach().cpu().numpy()  # [128,512]
    b_o = sd['linformer.seq.0.fn.w_o.bias'].detach().cpu().numpy()    # [128]

    # Norm 与 FFN 权重
    norm0_w = sd['linformer.seq.0.norm.weight'].detach().cpu().numpy()
    norm0_b = sd['linformer.seq.0.norm.bias'].detach().cpu().numpy()
    w1 = sd['linformer.seq.1.fn.w_1.weight'].detach().cpu().numpy()   # [512,128]
    b1 = sd['linformer.seq.1.fn.w_1.bias'].detach().cpu().numpy()     # [512]
    w2 = sd['linformer.seq.1.fn.w_2.weight'].detach().cpu().numpy()   # [128,512]
    b2 = sd['linformer.seq.1.fn.w_2.bias'].detach().cpu().numpy()     # [128]
    norm1_w = sd['linformer.seq.1.norm.weight'].detach().cpu().numpy()
    norm1_b = sd['linformer.seq.1.norm.bias'].detach().cpu().numpy()

    pos_slice = posEmb[:S] / 3.0
    x_in = x_emb + pos_slice

    # 计算 Q/K/V（两头）
    Q = []
    K = []
    V = []
    for h in range(2):
        Q.append(x_in @ to_q_w[h].T)  # [S,256]
        if args.precompress:
            Eh = E[h][:, :S]  # [32,S]
            Fh = F[h][:, :S]  # [32,S]
            X_E = Eh @ x_in           # [32,128]
            X_F = Fh @ x_in           # [32,128]
            K.append(X_E @ to_k_w[h].T)  # [32,256]
            V.append(X_F @ to_v_w[h].T)  # [32,256]
        else:
            K.append(x_in @ to_k_w[h].T)
            V.append(x_in @ to_v_w[h].T)

    # 低秩压缩：EK / FV
    EK = []
    FV = []
    for h in range(2):
        Eh = E[h][:, :S]  # [32,S]
        Fh = F[h][:, :S]  # [32,S]
        if args.precompress:
            # 已经在上面用E/F先压缩输入计算出K/V，这里无需再乘一次E/F
            EK.append(K[h])  # [32,256]
            FV.append(V[h])  # [32,256]
        else:
            EK.append(Eh @ K[h])  # [32,256]
            FV.append(Fh @ V[h])  # [32,256]

    # 注意力计算（压缩长度k=32）
    dim_d = Q[0].shape[-1]
    scale = 1.0 / np.sqrt(dim_d)
    O_heads = []
    for h in range(2):
        # logits: [S,32]
        logits = (Q[h] @ EK[h].T) * scale
        # softmax 按最后一维
        logits_max = logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits - logits_max)
        attn = exp / exp.sum(axis=-1, keepdims=True)  # [S,32]
        # 输出: [S,256]
        O = attn @ FV[h]
        O_heads.append(O)

    # 拼接头并线性输出
    O_cat = np.concatenate(O_heads, axis=-1)  # [S,512]
    attn_out = O_cat @ w_o.T + b_o.reshape(1, -1)  # [S,128]

    # 残差 + Norm0
    x_attn_res = x_in + attn_out
    x_norm0 = layernorm(x_attn_res, norm0_w, norm0_b)

    # 前馈 FFN: GELU(w1*x + b1) -> w2 + b2
    ff_hidden = gelu(x_norm0 @ w1.T + b1.reshape(1, -1))
    ff_out = ff_hidden @ w2.T + b2.reshape(1, -1)

    # 残差 + Norm1
    x_ff_res = x_norm0 + ff_out
    x_norm1 = layernorm(x_ff_res, norm1_w, norm1_b)

    m = np.ones((S,), dtype=np.int32)

    # x_max: 逐维取最大；x_mean: 有效位置平均
    x_max = x_norm1.max(axis=0)
    x_mean = (x_norm1 * m.reshape(-1, 1)).sum(axis=0) / (m.sum() if m.sum() > 0 else 1)
    x_feat = np.concatenate([x_max.reshape(1, -1), x_mean.reshape(1, -1)], axis=-1)  # [1,256]

    # 读取 fcLinear Sequential 权重
    # Linear(256->1024) + BN(1024) + ReLU + Dropout + Linear(1024->C)
    fc0_w = sd['fcLinear.0.weight'].detach().cpu().numpy()  # [1024,256]
    fc0_b = sd['fcLinear.0.bias'].detach().cpu().numpy()    # [1024]
    bn_w = sd['fcLinear.1.weight'].detach().cpu().numpy()   # [1024]
    bn_b = sd['fcLinear.1.bias'].detach().cpu().numpy()     # [1024]
    bn_rm = sd['fcLinear.1.running_mean'].detach().cpu().numpy()  # [1024]
    bn_rv = sd['fcLinear.1.running_var'].detach().cpu().numpy()   # [1024]
    fc4_w = sd['fcLinear.4.weight'].detach().cpu().numpy()  # [C,1024]
    fc4_b = sd['fcLinear.4.bias'].detach().cpu().numpy()    # [C]

    # 前向：fc0 -> BN(inference) -> ReLU -> fc4
    h0 = x_feat @ fc0_w.T + fc0_b.reshape(1, -1)  # [1,1024]
    eps_bn = 1e-5
    h_bn = ((h0 - bn_rm.reshape(1, -1)) / np.sqrt(bn_rv.reshape(1, -1) + eps_bn)) * bn_w.reshape(1, -1) + bn_b.reshape(1, -1)
    h_relu = np.maximum(h_bn, 0.0)
    y_logit = h_relu @ fc4_w.T + fc4_b.reshape(1, -1)  # [1,C]

    # 数值稳定softmax
    logits_max = y_logit.max(axis=-1, keepdims=True)
    y_prob = np.exp(y_logit - logits_max)
    y_prob = y_prob / y_prob.sum(axis=-1, keepdims=True)
    y_pred = int(np.argmax(y_prob, axis=-1)[0])

    print(f"The plaintext calculation result is: {y_pred}")


if __name__ == '__main__':
    main()
def load_input_vectors(dir_path: str, S: int) -> np.ndarray:
    vecs = []
    for i in range(S):
        fp = os.path.join(dir_path, f"input_{i}.txt")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"找不到输入文件: {fp}")
        v = np.loadtxt(fp, dtype=np.float32)
        vecs.append(v.reshape(1, -1))
    return np.vstack(vecs)