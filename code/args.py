# args.py
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Prototype Shifting Training Script")

    # === 模型与路径 ===
    # 增加了 qwen 选项
    parser.add_argument('--model_type', type=str, default="qwen", choices=["qwen"])
    parser.add_argument('--train_json', type=str, default=None)
    parser.add_argument('--test_json', type=str, default=None)
    parser.add_argument('--valid_json', type=str, default=None)
    parser.add_argument('--embed_base_path', type=str, default=os.environ.get("KDH_CAD_TMCAD_EMBEDDINGS"))

    # === 分类器配置 ===
    parser.add_argument('--classifier_mode', type=str, default="cross_attn", choices=["cross_attn"],
                        help="KDH-CAD classifier: image tokens attend to frozen textbook "
                             "prototypes; the shift vector acts only at the similarity stage.")

    # === 原型与更新策略 ===
    parser.add_argument('--shift_init_mode', type=str, default="data", choices=["random", "data"],
                        help="Initialization mode for shift_vector: 'random' or 'data' (visual centroids).")
    parser.add_argument('--use_shift', type=str, default="yes", choices=["yes", "no"])

    parser.add_argument('--use_shift_scale', type=str, default="no", choices=["yes", "no"],
                        help="Whether to use a learnable scaling factor for the shift vector.")

    # === 外部权重路径 ===
    parser.add_argument('--qwen_path', type=str, default=os.environ.get("KDH_CAD_QWEN_PATH"))
    
    parser.add_argument('--image_encode_mode', type=str, default="prompt")
    parser.add_argument('--pooling_method', type=str, default="mean")

    # === 训练超参 ===
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for trainable parts")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--dtype', type=str, default="bfloat16")

    # === SwanLab 配置 ===
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--swanlab_project', type=str, default="test")
    parser.add_argument('--swanlab_workspace', type=str, default="ANKIIMA")
    parser.add_argument('--swanlab_mode', type=str, default="cloud")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--best_model_path', type=str, default="best_model.pth")
    parser.add_argument('--skip_test', type=str, default="no", choices=["yes", "no"],
                        help="Skip final test-set evaluation after training. Validation and best-checkpoint selection still run.")
    parser.add_argument('--output_json', type=str, default="semantic_output_results.json")
    parser.add_argument('--max_samples', type=int, default=0)

    args = parser.parse_args()

    required = ("train_json", "test_json", "valid_json", "embed_base_path", "qwen_path")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("Missing required paths: " + ", ".join("--" + name for name in missing))

    # 自动转换 dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    args.dtype = dtype_map.get(args.dtype, torch.float32)

    return args
