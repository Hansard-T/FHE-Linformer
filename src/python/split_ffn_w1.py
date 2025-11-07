import os
import argparse
import numpy as np
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_w1_from_ckpt(ckpt_path: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    w = sd['linformer.transformerLayers.transformer0.ffn.Wffn.0.weight'].detach().cpu().numpy()
    return w


def load_w1_from_txt(weight_txt_path: str, expect_shape=(512, 128)) -> np.ndarray:
    arr = np.loadtxt(weight_txt_path, delimiter=',')
    if arr.ndim == 1:
        total = arr.size
        rows, cols = expect_shape
        if total != rows * cols:
            raise ValueError(f"文本总元素数 {total} 不等于期望 {rows*cols}")
        arr = arr.reshape(rows, cols)
    elif tuple(arr.shape) != expect_shape:
        raise ValueError(f"文本矩阵形状 {arr.shape} 不等于期望 {expect_shape}")
    return arr.astype(np.float32)


def split_into_blocks(W: np.ndarray, rows_per_block: int = 128) -> list:
    rows, cols = W.shape
    if rows % rows_per_block != 0:
        raise ValueError(f"行数 {rows} 不能被每块行数 {rows_per_block} 整除")
    blocks = []
    for i in range(0, rows, rows_per_block):
        blocks.append(W[i:i+rows_per_block, :])
    return blocks


def save_blocks(blocks: list, out_dir: str, base_name: str = 'ffn_W1_block', fmt: str = '%.18e', delimiter: str = ','):
    ensure_dir(out_dir)
    for idx, B in enumerate(blocks):
        fp = os.path.join(out_dir, f"{base_name}_{idx}.txt")
        np.savetxt(fp, B, fmt=fmt, delimiter=delimiter)
    return [os.path.join(out_dir, f"{base_name}_{i}.txt") for i in range(len(blocks))]


def main():
    parser = argparse.ArgumentParser(description='将 FFN 第一层权重 (512×128) 拆分为四个 128×128 文件')
    parser.add_argument('--weight_txt', default='../../weights-20NG/linformer_transformerLayers_transformer0_ffn_Wffn_0_weight.txt', help='数值参数文件路径（例如 linformer_transformerLayers_transformer0_ffn_Wffn_0_weight.txt）')
    parser.add_argument('--out_dir', default='../../weights-20NG', help='输出目录')
    parser.add_argument('--rows_per_block', type=int, default=128, help='每块的行数，默认128')
    parser.add_argument('--fmt', default='%.18e', help='保存格式，默认 %.18e')
    parser.add_argument('--delimiter', default=',', help='保存分隔符，默认 ,')
    args = parser.parse_args()

    W = load_w1_from_txt(args.weight_txt, expect_shape=(args.rows_per_block * 4, 128))

    if W.shape != (args.rows_per_block * 4, 128):
        raise ValueError(f"权重形状 {W.shape} 与期望 {(args.rows_per_block * 4, 128)} 不一致")

    blocks = split_into_blocks(W, rows_per_block=args.rows_per_block)
    paths = save_blocks(blocks, args.out_dir, base_name='ffn_W1_block', fmt=args.fmt, delimiter=args.delimiter)
    print('保存以下文件:')
    for p in paths:
        print(p)

    os.system('rm ../../weights-20NG/linformer_transformerLayers_transformer0_ffn_Wffn_0_weight.txt')


if __name__ == '__main__':
    main()