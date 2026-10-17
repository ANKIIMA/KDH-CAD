import os
import json
import math
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from loguru import logger

# OCC 相关导入
try:
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.gp import gp_Dir
    from OCC.Extend.DataExchange import read_step_file
    from OCC.Display.OCCViewer import Viewer3d
except ImportError:
    logger.warning("OCC modules not found. Rendering will fail but JSON logic can be tested.")

# ================= 静态配置 =================
RAW_TO_TARGET_MAP = {
    "Washers": "Washers",
    "O_Rings": "O_Rings",
    "Bushing": "Bushing",
    "Shaft_Collar": "Shaft_Collar",
    "Grommets": "Grommets",
    "Keyway_Shaft": "Keyway_Shaft",
    "Rotary_Shaft": "Rotary_Shaft",
    "Socket_Head_Screws": "Socket_Head_Screws"
}

TARGET_CLASSES = [
    "Washers", "O_Rings", "Bushing", "Shaft_Collar", 
    "Grommets", "Keyway_Shaft", "Rotary_Shaft", "Socket_Head_Screws"
]

# ================= 渲染核心逻辑 (Worker 函数) =================
def render_worker(item_info):
    """单进程渲染逻辑"""
    step_path, raw_cls, target_cls, img_root, num_views, class_index = item_info
    os.makedirs(img_root, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(step_path))[0]
    
    expected_paths = [
        os.path.join(img_root, f"{target_cls}_{base_name}_view_{i}.png")
        for i in range(num_views)
    ]
    
    if all(os.path.exists(p) for p in expected_paths):
        return {
            "part_key": f"{raw_cls}_{base_name}",
            "label": class_index,
            "label_name": target_cls,
            "image_paths": expected_paths,
            "num_views": num_views,
            "step_path": step_path
        }

    success = True
    try:
        offscreen_renderer = Viewer3d()
        offscreen_renderer.Create()
        offscreen_renderer.SetModeShaded()
        offscreen_renderer.View.SetBackgroundColor(Quantity_Color(1, 1, 1, Quantity_TOC_RGB))
        
        shape = read_step_file(step_path)
        if shape is None or shape.IsNull():
            success = False
        else:
            offscreen_renderer.DisplayShape(shape, update=True)
            offscreen_renderer.FitAll()

            golden_angle = math.pi * (3 - math.sqrt(5))
            for i in range(num_views):
                z = 1 - (i / (num_views - 1)) * 2 if num_views > 1 else 1
                radius = math.sqrt(1 - z * z)
                theta = golden_angle * i
                x, y = math.cos(theta) * radius, math.sin(theta) * radius
                
                offscreen_renderer.View.SetProj(x, y, z)
                view_dir = gp_Dir(x, y, z)
                up_dir = gp_Dir(0, 0, 1) if abs(view_dir.Dot(gp_Dir(0, 1, 0))) > 0.99 else gp_Dir(0, 1, 0)
                offscreen_renderer.View.SetUp(up_dir.X(), up_dir.Y(), up_dir.Z())
                
                offscreen_renderer.FitAll()
                offscreen_renderer.View.Dump(expected_paths[i])
            offscreen_renderer.EraseAll()
    except Exception as e:
        logger.error(f"渲染异常 {step_path}: {e}")
        success = False
    finally:
        try: del offscreen_renderer
        except: pass

    if not success: return None
    return {
        "part_key": f"{raw_cls}_{base_name}",
        "label": class_index,
        "label_name": target_cls,
        "image_paths": expected_paths,
        "num_views": num_views,
        "step_path": step_path
    }

# ================= 主处理流程 =================
def main():
    parser = argparse.ArgumentParser(description="FebWave 数据集划分（固定测试集+随机训练集）")
    parser.add_argument("--src_root", type=str, required=True, help="源文件根目录")
    parser.add_argument("--img_root", type=str, required=True, help="图片保存目录")
    parser.add_argument("--json_dir", type=str, required=True, help="JSON标注保存目录")
    parser.add_argument("--n_train", type=int, default=50, help="每类训练集数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--split_mode", type=str, 
                        choices=["tail_fixed_random", "tail_fixed_sequential"], 
                        default="tail_fixed_random",
                        help="tail_fixed_random: 最后100个固定，前面随机选n个训练")
    parser.add_argument("--num_workers", type=int, default=8, help="并行进程数")
    args = parser.parse_args()

    os.makedirs(args.json_dir, exist_ok=True)

    # 1. 扫描文件
    logger.info(f"正在扫描: {args.src_root}")
    samples_by_target = defaultdict(list)
    for raw_cls, target_cls in RAW_TO_TARGET_MAP.items():
        cls_dir = os.path.join(args.src_root, raw_cls)
        if not os.path.isdir(cls_dir): continue
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith(('.stp', '.step')):
                    samples_by_target[target_cls].append({"step_path": os.path.join(root, f), "raw_cls": raw_cls})
        logger.info(f"类别 '{target_cls}': 扫描到 {len(samples_by_target[target_cls])} 个文件")

    # 2. 执行划分逻辑
    splits = {"train": [], "valid": [], "test": []}
    logger.info(f"正在执行模式: {args.split_mode} (Seed: {args.seed})")

    for target_cls in TARGET_CLASSES:
        all_samples = samples_by_target[target_cls]
        total = len(all_samples)
        
        # 【核心步骤 1】首先按路径排序，确保基础顺序绝对固定
        all_samples.sort(key=lambda x: x["step_path"])

        if total < 100 + args.n_train:
            logger.warning(f"类别 {target_cls} 样本不足({total})，跳过该类。")
            continue

        # 【核心步骤 2】固定取最后 100 个作为 Val (前50) 和 Test (后50)
        # 这部分不受 seed 影响
        c_val = all_samples[-100:-50]
        c_test = all_samples[-50:]

        # 【核心步骤 3】处理训练集：从排除掉最后100个后的剩余样本中选择
        head_pool = all_samples[:-100]
        
        if args.split_mode == "tail_fixed_random":
            # 仅对头部池进行随机打乱
            random.seed(args.seed)
            random.shuffle(head_pool)
            c_train = head_pool[:args.n_train]
        else:
            # 顺序选取（tail_fixed_sequential）
            c_train = head_pool[:args.n_train]

        for s in c_train: splits["train"].append((s, target_cls))
        for s in c_val: splits["valid"].append((s, target_cls))
        for s in c_test: splits["test"].append((s, target_cls))

    # 3. 并行渲染处理
    for split_name, data_list in splits.items():
        if not data_list:
            logger.info(f"跳过空数据集划分: {split_name}")
            continue
        
        logger.info(f"开始处理 {split_name} (共 {len(data_list)} 个样本)...")
        # 任务列表
        tasks = [(s["step_path"], s["raw_cls"], tc, args.img_root, 12, TARGET_CLASSES.index(tc)) for s, tc in data_list]
        
        final_results = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # 使用列表保存 futures 以维持提交顺序
            futures = [executor.submit(render_worker, t) for t in tasks]
            for f in tqdm(futures, total=len(futures), desc=f"处理 {split_name}"):
                res = f.result()
                if res: final_results.append(res)

        # 【重要】在保存 JSON 前再次按路径排序，确保 JSON 文件的行顺序在不同 seed 下也是一致的（方便对比）
        final_results.sort(key=lambda x: x["step_path"])

        output_json = os.path.join(args.json_dir, f"{split_name}.json")
        with open(output_json, "w") as f:
            json.dump(final_results, f, indent=4)
        logger.success(f"完成！{split_name} 标注已存至: {output_json}")

if __name__ == "__main__":
    main()