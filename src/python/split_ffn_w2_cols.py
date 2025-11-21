import argparse
import os
import numpy as np


def load_w_from_txt(weight_txt_path: str, expect_shape=(128, 512)) -> np.ndarray:
    try:
        arr = np.loadtxt(weight_txt_path, delimiter=',')
    except Exception:
        arr = np.loadtxt(weight_txt_path)
    if arr.ndim == 1:
        total = arr.size
        rows, cols = expect_shape
        if total != rows * cols:
            raise ValueError(f"文本总元素数 {total} 不等于期望 {rows*cols}")
        arr = arr.reshape(rows, cols)
    elif tuple(arr.shape) != expect_shape:
        raise ValueError(f"文本矩阵形状 {arr.shape} 不等于期望 {expect_shape}")
    return arr.astype(np.float32)


def split_col_blocks(W: np.ndarray, cols_per_block: int = 128) -> list:
    total_cols = W.shape[1]
    if total_cols % cols_per_block != 0:
        raise ValueError(f"列数 {total_cols} 不能被每块列数 {cols_per_block} 整除")
    blocks = []
    for i in range(0, total_cols, cols_per_block):
        blocks.append(W[:, i:i + cols_per_block])
    return blocks


def save_blocks(blocks: list, out_dir: str, base_name: str = 'ffn_W2_block', fmt: str = '%.18e', delimiter: str = ','):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for idx, B in enumerate(blocks):
        fp = os.path.join(out_dir, f"{base_name}_{idx}.txt")
        np.savetxt(fp, B, fmt=fmt, delimiter=delimiter)
        paths.append(fp)
    return paths


def main():
    parser = argparse.ArgumentParser(description='将 128×512 权重矩阵按列切成 4 个 128×128 文件')
    parser.add_argument('--weight_txt', default='weights-20NG/linformer_transformerLayers_transformer0_ffn_Wffn_2_weight.txt')
    parser.add_argument('--out_dir', default='weights-20NG')
    parser.add_argument('--cols_per_block', type=int, default=128)
    args = parser.parse_args()

    W = load_w_from_txt(args.weight_txt, expect_shape=(128, 512))
    blocks = split_col_blocks(W, cols_per_block=args.cols_per_block)
    paths = save_blocks(blocks, args.out_dir, base_name='ffn_W2_block')

    for p in paths:
        print(p)


if __name__ == '__main__':
    main()