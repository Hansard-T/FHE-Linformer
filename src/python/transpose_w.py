import argparse
import os
try:
    import numpy as np
    HAS_NP = True
except Exception:
    np = None
    HAS_NP = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default='weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WV_weight.txt')
    p.add_argument('--out', dest='out_path', default='weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WV_weight_T.txt')
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--cols', type=int, default=128)
    p.add_argument('--delimiter', default=',')
    p.add_argument('--fmt', default='%.18e')
    a = p.parse_args()

    if HAS_NP:
        try:
            arr = np.loadtxt(a.in_path, delimiter=a.delimiter)
        except Exception:
            arr = np.loadtxt(a.in_path)

        if arr.ndim == 1:
            total = arr.size
            if total != a.rows * a.cols:
                raise SystemExit(f'元素数 {total} 不等于 {a.rows*a.cols}')
            arr = arr.reshape(a.rows, a.cols)
        elif arr.shape != (a.rows, a.cols):
            raise SystemExit(f'矩阵形状 {arr.shape} 不等于 {(a.rows, a.cols)}')

        arr_t = arr.T
        os.makedirs(os.path.dirname(a.out_path) or '.', exist_ok=True)
        np.savetxt(a.out_path, arr_t, fmt=a.fmt, delimiter=a.delimiter)
        print(a.out_path)
        print(arr_t.shape)
    else:
        with open(a.in_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        mat = []
        for ln in lines:
            if a.delimiter:
                toks = [t for t in ln.split(a.delimiter) if t != '']
            else:
                toks = ln.split()
            row = [float(t) for t in toks]
            mat.append(row)

        if len(mat) == 1 and len(mat[0]) == a.rows * a.cols:
            flat = mat[0]
            mat = [flat[i*a.cols:(i+1)*a.cols] for i in range(a.rows)]

        if len(mat) != a.rows:
            raise SystemExit(f'行数 {len(mat)} 不等于 {a.rows}')
        for i, row in enumerate(mat):
            if len(row) != a.cols:
                raise SystemExit(f'第{i}行列数 {len(row)} 不等于 {a.cols}')

        mat_t = list(map(list, zip(*mat)))

        os.makedirs(os.path.dirname(a.out_path) or '.', exist_ok=True)
        with open(a.out_path, 'w') as f:
            for row in mat_t:
                line = a.delimiter.join([a.fmt % x for x in row])
                f.write(line + '\n')
        print(a.out_path)
        print((len(mat_t), len(mat_t[0]) if mat_t else 0))

if __name__ == '__main__':
    main()
