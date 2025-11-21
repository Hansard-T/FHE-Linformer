import os
import torch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "weights-sst2")
os.makedirs(OUT_DIR, exist_ok=True)

def save_vector_txt(path, tensor):
    arr = tensor.detach().cpu().numpy().reshape(-1)
    with open(path, "w") as f:
        for v in arr:
            f.write(f"{float(v)}\n")

def save_matrix_txt(path, tensor):
    # 默认按行展平；若需要转置可改为 tensor.t()
    arr = tensor.detach().cpu().numpy().reshape(-1)
    with open(path, "w") as f:
        for v in arr:
            f.write(f"{float(v)}\n")

def split_rows_4(weight):  # (512,128) -> 4*(128,128)
    assert weight.shape[0] == 512 and weight.shape[1] == 128
    chunks = torch.chunk(weight, 4, dim=0)
    return chunks  # list of 4 tensors (128,128)

def split_cols_4(weight):  # (128,512) -> 4*(128,128)
    assert weight.shape[0] == 128 and weight.shape[1] == 512
    chunks = torch.chunk(weight, 4, dim=1)
    return chunks  # list of 4 tensors (128,128)

def export_from_bin(bin_path: str):
    obj = torch.load(bin_path, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    keys = list(sd.keys())
    has_bert_prefix = any(k.startswith("bert.") for k in keys)
    base = "bert." if has_bert_prefix else ""

    pw_key = base + "pooler.dense.weight"
    pb_key = base + "pooler.dense.bias"
    if pw_key in sd and pb_key in sd:
        save_matrix_txt(os.path.join(OUT_DIR, "pooler_dense_weight.txt"), sd[pw_key])
        save_vector_txt(os.path.join(OUT_DIR, "pooler_dense_bias.txt"), sd[pb_key])

    max_layer = -1
    for k in keys:
        if k.startswith(base + "encoder.layer."):
            try:
                idx = int(k[len(base + "encoder.layer."):].split(".")[0])
                if idx > max_layer:
                    max_layer = idx
            except Exception:
                pass

    for i in range(max_layer + 1):
        prefix = f"{base}encoder.layer.{i}"
        save_matrix_txt(os.path.join(OUT_DIR, f"layer{i}_attself_query_weight.txt"), sd[f"{prefix}.attention.self.query.weight"])
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_attself_query_bias.txt"), sd[f"{prefix}.attention.self.query.bias"])
        save_matrix_txt(os.path.join(OUT_DIR, f"layer{i}_attself_key_weight.txt"), sd[f"{prefix}.attention.self.key.weight"])
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_attself_key_bias.txt"), sd[f"{prefix}.attention.self.key.bias"])
        save_matrix_txt(os.path.join(OUT_DIR, f"layer{i}_attself_value_weight.txt"), sd[f"{prefix}.attention.self.value.weight"])
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_attself_value_bias.txt"), sd[f"{prefix}.attention.self.value.bias"])
        save_matrix_txt(os.path.join(OUT_DIR, f"layer{i}_selfoutput_weight.txt"), sd[f"{prefix}.attention.output.dense.weight"])
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_selfoutput_bias.txt"), sd[f"{prefix}.attention.output.dense.bias"])
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_selfoutput_vy.txt"), sd[f"{prefix}.attention.output.LayerNorm.weight"])
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_selfoutput_normbias.txt"), sd[f"{prefix}.attention.output.LayerNorm.bias"])
        with open(os.path.join(OUT_DIR, f"layer{i}_selfoutput_mean.txt"), "w") as f:
            for _ in range(sd[f"{prefix}.attention.output.LayerNorm.weight"].shape[0]):
                f.write("0.0\n")

        inter_w = sd[f"{prefix}.intermediate.dense.weight"]
        inter_b = sd[f"{prefix}.intermediate.dense.bias"]
        chunks = split_rows_4(inter_w)
        for k, ck in enumerate(chunks, 1):
            save_matrix_txt(os.path.join(OUT_DIR, f"layer{i}_intermediate_weight{k}.txt"), ck)
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_intermediate_bias.txt"), inter_b)

        out_w = sd[f"{prefix}.output.dense.weight"]
        out_b = sd[f"{prefix}.output.dense.bias"]
        chunks = split_cols_4(out_w)
        for k, ck in enumerate(chunks, 1):
            save_matrix_txt(os.path.join(OUT_DIR, f"layer{i}_output_weight{k}.txt"), ck)
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_output_bias.txt"), out_b)
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_output_vy.txt"), sd[f"{prefix}.output.LayerNorm.weight"])
        save_vector_txt(os.path.join(OUT_DIR, f"layer{i}_output_normbias.txt"), sd[f"{prefix}.output.LayerNorm.bias"])
        with open(os.path.join(OUT_DIR, f"layer{i}_output_mean.txt"), "w") as f:
            for _ in range(sd[f"{prefix}.output.LayerNorm.weight"].shape[0]):
                f.write("0.0\n")

def export_classifier_from_sd(sd: dict):
    for k in ["classifier.weight", "bert.classifier.weight"]:
        if k in sd:
            save_matrix_txt(os.path.join(OUT_DIR, "classifier_weight.txt"), sd[k])
            break
    for k in ["classifier.bias", "bert.classifier.bias"]:
        if k in sd:
            save_vector_txt(os.path.join(OUT_DIR, "classifier_bias.txt"), sd[k])
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..", "trained_models", "SST-2-BERT-Tiny.bin"))
    args = parser.parse_args()
    obj = torch.load(args.bin, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    export_from_bin(args.bin)
    export_classifier_from_sd(sd)
    print("Export completed:", OUT_DIR)