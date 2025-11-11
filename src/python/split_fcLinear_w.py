import os
import argparse
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_fc_from_txt(weight_txt_path: str, expect_shape=(1024, 128), delimiter: str = ',') -> np.ndarray:
    """
    读取 `fcLinear_0_weight.txt`。
    - 支持逗号分隔的矩阵文本（1024 行 × 128 列）或扁平 131072 个数的一行文本。
    - 返回形状 (1024, 128) 的 numpy 数组。
    """
    arr = np.loadtxt(weight_txt_path, delimiter=delimiter)
    if arr.ndim == 1:
        total = arr.size
        rows, cols = expect_shape
        if total != rows * cols:
            raise ValueError(f"文本总元素数 {total} 不等于期望 {rows * cols}")
        arr = arr.reshape(rows, cols)
    elif tuple(arr.shape) != expect_shape:
        raise ValueError(f"文本矩阵形状 {arr.shape} 不等于期望 {expect_shape}")
    return arr.astype(np.float64)


def split_vertical(W: np.ndarray, tile_cols: int = 128) -> list:
    """
    按“竖着读取”的编码方式拆分：
    1) 先转置为 (128, 1024)，让行索引 j 对应输入维度 j；
    2) 再按列将 1024 列切成 8 个 128 列的块，得到 8 个 (128, 128) 矩阵。
    """
    WT = W.T  # (128, 1024)
    rows, cols = WT.shape
    if rows != 128:
        raise ValueError(f"转置后行数应为 128，实际 {rows}")
    if cols % tile_cols != 0:
        raise ValueError(f"列数 {cols} 不能被每块列数 {tile_cols} 整除")
    blocks = []
    for i in range(0, cols, tile_cols):
        blocks.append(WT[:, i:i + tile_cols])  # (128, 128)
    return blocks


def save_blocks(blocks: list, out_dir: str, base_name: str = 'fcLinear_0_weight_block', fmt: str = '%.18e', delimiter: str = ','):
    ensure_dir(out_dir)
    paths = []
    for idx, B in enumerate(blocks):
        fp = os.path.join(out_dir, f"{base_name}_{idx}.txt")
        np.savetxt(fp, B, fmt=fmt, delimiter=delimiter)
        paths.append(fp)
    return paths


def main():
    parser = argparse.ArgumentParser(description='将 fcLinear_0_weight (1024×128) 先转置为 128×1024，再按列拆分为 8 个 128×128 文件')
    parser.add_argument('--weight_txt', default='../../weights-20NG/fcLinear_0_weight.txt', help='输入权重文本路径 (1024×128 或扁平 131072)')
    parser.add_argument('--out_dir', default='../../weights-20NG', help='输出目录')
    parser.add_argument('--tile_cols', type=int, default=128, help='每个块的列数，默认 128')
    parser.add_argument('--fmt', default='%.18e', help='保存格式，默认 %.18e')
    parser.add_argument('--delimiter', default=',', help='保存分隔符，默认 ,')
    args = parser.parse_args()

    W = load_fc_from_txt(args.weight_txt, expect_shape=(args.tile_cols * 8, 128), delimiter=args.delimiter)

    blocks = split_vertical(W, tile_cols=args.tile_cols)
    paths = save_blocks(blocks, args.out_dir, base_name='fcLinear_0_weight_block', fmt=args.fmt, delimiter=args.delimiter)

    print('保存以下文件:')
    for p in paths:
        print(p)


if __name__ == '__main__':
    main()