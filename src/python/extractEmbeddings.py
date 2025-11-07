import os
import argparse
import numpy as np
import torch
import pickle

from tokenizerFuncs import Tokenizer
from utils import Sklearn_20NG

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_tokenizer_for_20ng(data_root: str, seq_max_len: int = 700):
    # 用全部 train+test 序列初始化 Tokenizer，重建与训练时一致的词表
    trainDS = Sklearn_20NG(data_root, 'train')
    testDS = Sklearn_20NG(data_root, 'test')
    tokenizer = Tokenizer(trainDS.sequences + testDS.sequences,
                          trainDS.labels + testDS.labels,
                          seqMaxLen=seq_max_len)
    return tokenizer

def load_embedding_from_cache(cache_path: str) -> np.ndarray:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"找不到缓存文件: {cache_path}")
    # vectorize 使用 pickle.dump 保存 numpy 数组，这里使用 pickle.load 读取
    with open(cache_path, 'rb') as f:
        embedding = pickle.load(f)
    if not isinstance(embedding, np.ndarray):
        raise ValueError("缓存文件未包含numpy数组embedding")
    return embedding.astype(np.float32)

def load_embedding_from_ckpt(ckpt_path: str) -> np.ndarray:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到检查点: {ckpt_path}")
    model_state = torch.load(ckpt_path, map_location='cpu')
    # 兼容 BaseModel.save 保存的字典结构
    if isinstance(model_state, dict) and 'model_state' in model_state:
        sd = model_state['model_state']
    elif isinstance(model_state, dict):
        sd = model_state
    else:
        raise ValueError("检查点格式不符合预期")

    key_candidates = [
        'embedding.embedding.weight',
        'sEmbedding.embedding.weight',
        'tEmbedding.embedding.weight',
    ]
    emb = None
    for k in key_candidates:
        if k in sd and isinstance(sd[k], torch.Tensor):
            emb = sd[k].detach().cpu().numpy()
            break
    if emb is None:
        for k, v in sd.items():
            if k.endswith('embedding.weight') and isinstance(v, torch.Tensor):
                emb = v.detach().cpu().numpy()
                break
    if emb is None:
        raise KeyError("在检查点中未找到embedding权重")
    return emb.astype(np.float32)

def get_sample_tokens(data_root: str, split: str, index: int):
    ds = Sklearn_20NG(data_root, split)
    if index < 0 or index >= len(ds.sequences):
        raise IndexError(f"索引越界: {index}, 有效范围 0..{len(ds.sequences)-1}")
    return ds.sequences[index]

def save_sample_embeddings(vectors: np.ndarray, tokens: list, out_dir: str):
    ensure_dir(out_dir)
    # 保存每个 token 的向量到独立文件，方便与示例脚本保持一致
    for i in range(len(tokens)):
        np.savetxt(os.path.join(out_dir, f'input_{i}.txt'), vectors[i], delimiter=',')

def main():
    parser = argparse.ArgumentParser(description='提取20NG单条样本的词向量并保存到文件')
    parser.add_argument('--data_root', default='/Users/tangxianning/Downloads/FHE-Linformer/datasets/20NG', help='20NG数据根目录')
    parser.add_argument('--split', choices=['train', 'test'], default='train', help='选择数据集划分')
    parser.add_argument('--index', type=int, default=1, help='样本索引')
    parser.add_argument('--source', choices=['cache', 'ckpt'], default='cache', help='向量来源：缓存或检查点')
    parser.add_argument('--cache_path', default='/Users/tangxianning/Downloads/FHE-Linformer/cache/skipgram_d128_NG.pkl', help='当source=cache时的缓存路径')
    parser.add_argument('--ckpt_path', default='', help='当source=ckpt时的检查点路径')
    parser.add_argument('--save_dir', default='/Users/tangxianning/Downloads/FHE-Linformer/src/tmp_embeddings', help='输出保存目录')
    args = parser.parse_args()

    # 构建一致词表的 Tokenizer
    tokenizer = load_tokenizer_for_20ng(args.data_root)
    # 取指定样本的 token 序列
    sample_tokens = get_sample_tokens(args.data_root, args.split, args.index)

    # 将该样本映射为 ids（eval=False 仅按样本长度，不做全局 padding）
    ids_list, mask_list = tokenizer.tokenize_sequences([sample_tokens], eval=False)
    ids = np.array(ids_list[0], dtype=np.int64)
    mask = np.array(mask_list[0], dtype=np.bool_)  # True=有效，False=PAD

    # 加载词向量矩阵
    if args.source == 'cache':
        embedding_matrix = load_embedding_from_cache(args.cache_path)
    else:
        if not args.ckpt_path:
            raise ValueError('当source=ckpt时必须提供--ckpt_path')
        embedding_matrix = load_embedding_from_ckpt(args.ckpt_path)

    if embedding_matrix.shape[0] <= ids.max():
        raise ValueError(
            f"embedding矩阵行数({embedding_matrix.shape[0]})不足以索引最大id({int(ids.max())})，"
            "请确认词表与embedding来源一致。"
        )

    # 查表得到该样本的 [S, D] 词向量序列
    vectors = embedding_matrix[ids]
    # 避免潜在的PAD（正常情况下eval=False不会产生PAD）
    vectors_valid = vectors[mask]
    tokens_valid = [t for t, m in zip(sample_tokens, mask) if m]

    save_sample_embeddings(vectors_valid, tokens_valid, args.save_dir)
    print(f"已保存 {len(tokens_valid)} 个token的向量到目录：{args.save_dir}")


if __name__ == '__main__':
    main()