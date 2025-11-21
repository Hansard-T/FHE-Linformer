import os
import argparse
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_w_from_txt(weight_txt_path: str, expect_shape=(512, 128)) -> np.ndarray:
    """读取 txt 权重文件并检查形状"""
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


def split_transposed_blocks(W: np.ndarray, cols_per_block: int = 128) -> list:
    """
    W: 原始 512×128
    转置 -> 128×512
    按列切块，每块 128 列 -> 得到 4 个 128×128 矩阵
    """
    W_T = W.T  # 128×512
    total_cols = W_T.shape[1]  # 512
    if total_cols % cols_per_block != 0:
        raise ValueError(f"列数 {total_cols} 不能被每块列数 {cols_per_block} 整除")
    blocks = []
    for i in range(0, total_cols, cols_per_block):
        blocks.append(W_T[:, i:i+cols_per_block])  # 128×128
    return blocks


def save_blocks(blocks: list, out_dir: str, base_name: str = 'ffn_W_block', fmt: str = '%.18e', delimiter: str = ','):
    ensure_dir(out_dir)
    paths = []
    for idx, B in enumerate(blocks):
        fp = os.path.join(out_dir, f"{base_name}_{idx}.txt")
        np.savetxt(fp, B, fmt=fmt, delimiter=delimiter)
        paths.append(fp)
    return paths


def main():
    parser = argparse.ArgumentParser(description='将 512×128 权重矩阵转置为 128×512 并切成 4 个 128×128 文件')
    parser.add_argument('--weight_txt', required=True, help='输入权重 txt 文件路径')
    parser.add_argument('--out_dir', default='../../weights-20NG', help='输出目录')
    parser.add_argument('--cols_per_block', type=int, default=128, help='每块列数')
    args = parser.parse_args()

    W = load_w_from_txt(args.weight_txt, expect_shape=(512, 128))
    blocks = split_transposed_blocks(W, cols_per_block=args.cols_per_block)
    paths = save_blocks(blocks, args.out_dir, base_name='ffn_W2_transposed_block')

    print("保存以下文件:")
    for p in paths:
        print(p)


if __name__ == '__main__':
    main()