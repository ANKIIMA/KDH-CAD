import os
import json
import math
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# OCC 相关导入
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.gp import gp_Dir
from OCC.Extend.DataExchange import read_step_file
from OCC.Display.OCCViewer import Viewer3d

# ================= 静态配置 =================
RAW_TO_TARGET_MAP = {
    "bearing": "bearing", "bolt": "bolt_screw", "screw": "bolt_screw",
    "bracket": "bracket", "coupling": "coupling", "flange": "flange",
    "gear": "gear", "nut": "nut", "pulley": "pulley",
    "shaft": "shaft", "spring": "spring"
}

TARGET_CLASSES = [
    "bearing", "bolt_screw", "bracket", "coupling", "flange", 
    "gear", "nut", "pulley", "shaft", "spring"
]

# ================= 渲染核心逻辑 (Worker 函数) =================
def render_worker(item_info):
    """
    单个进程执行的任务：渲染一个零件的多个视角图片并返回该样本的元数据字典。
    """
    step_path, raw_cls, target_cls, img_root, num_views, class_index = item_info
    
    os.makedirs(img_root, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(step_path))[0]
    
    # 预定义视角图片的保存路径
    expected_paths = [
        os.path.join(img_root, f"{target_cls}_{base_name}_view_{i}.png")
        for i in range(num_views)
    ]
    
    success = True
    # 如果图片已存在则跳过渲染过程
    if not all(os.path.exists(p) for p in expected_paths):
        try:
            # 初始化独立的渲染器实例
            offscreen_renderer = Viewer3d()
            offscreen_renderer.Create()
            offscreen_renderer.SetModeShaded()
            offscreen_renderer.View.SetBackgroundColor(Quantity_Color(1, 1, 1, Quantity_TOC_RGB)) # 白底
            
            shape = read_step_file(step_path)
            if shape is None or shape.IsNull():
                success = False
            else:
                offscreen_renderer.DisplayShape(shape, update=True)
                offscreen_renderer.FitAll()

                # 黄金角度采样逻辑 (Fibonacci Sphere)
                # 计算公式: $z_i = 1 - \frac{i}{N-1} \times 2$
                golden_angle = math.pi * (3 - math.sqrt(5))
                for i in range(num_views):
                    z = 1 - (i / (num_views - 1)) * 2 if num_views > 1 else 1
                    radius = math.sqrt(1 - z * z)
                    theta = golden_angle * i
                    x, y = math.cos(theta) * radius, math.sin(theta) * radius
                    
                    # 设置相机位置和方向
                    offscreen_renderer.View.SetProj(x, y, z)
                    view_dir = gp_Dir(x, y, z)
                    # 处理 Up 方向，防止相机翻转
                    up_dir = gp_Dir(0, 0, 1) if abs(view_dir.Dot(gp_Dir(0, 1, 0))) > 0.99 else gp_Dir(0, 1, 0)
                    offscreen_renderer.View.SetUp(up_dir.X(), up_dir.Y(), up_dir.Z())
                    
                    offscreen_renderer.FitAll()
                    offscreen_renderer.View.Dump(expected_paths[i])
                
                offscreen_renderer.EraseAll()
        except Exception as e:
            print(f"\n[Error] 渲染失败 {step_path}: {e}")
            success = False
        finally:
            try:
                del offscreen_renderer
            except:
                pass

    if not success:
        return None

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
    parser = argparse.ArgumentParser(description="TMCAD 多视图数据集并行处理脚本")
    parser.add_argument("--src_root", type=str, required=True, help="STEP文件源根目录")
    parser.add_argument("--img_root", type=str, required=True, help="生成的图片保存路径")
    parser.add_argument("--json_dir", type=str, required=True, help="生成的JSON标注文件保存目录")
    parser.add_argument("--n_train", type=int, default=50, help="每个类别的训练集数量")
    parser.add_argument("--n_val", type=int, default=50, help="每个类别的验证集数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（仅在 random 模式下生效）")
    parser.add_argument("--split_mode", type=str, choices=["sequential", "random"], default="random", 
                        help="划分模式：sequential (取前N个) 或 random (使用seed随机选)")
    parser.add_argument("--num_workers", type=int, default=8, help="并行进程数")
    args = parser.parse_args()

    os.makedirs(args.json_dir, exist_ok=True)

    # 1. 扫描物理文件并按目标类别分类
    samples_by_target = defaultdict(list)
    for raw_cls, target_cls in RAW_TO_TARGET_MAP.items():
        cls_dir = os.path.join(args.src_root, raw_cls)
        if not os.path.isdir(cls_dir): continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.stp', '.step')):
                samples_by_target[target_cls].append({
                    "step_path": os.path.join(cls_dir, f), 
                    "raw_cls": raw_cls
                })

    # 2. 执行划分逻辑
    splits = {"train": [], "valid": [], "test": []}
    
    for target_cls in TARGET_CLASSES:
        all_samples = samples_by_target[target_cls]
        
        if args.split_mode == "random":
            # 随机模式：使用指定的 seed 进行洗牌
            random.seed(args.seed)
            random.shuffle(all_samples)
        else:
            # 顺序模式：按文件名排序，确保“前几个”是固定的
            all_samples.sort(key=lambda x: x["step_path"])
        
        c_train = all_samples[:args.n_train]
        c_val = all_samples[args.n_train : args.n_train + args.n_val]
        c_test = all_samples[args.n_train + args.n_val :]
        
        for s in c_train: splits["train"].append((s, target_cls))
        for s in c_val: splits["valid"].append((s, target_cls))
        for s in c_test: splits["test"].append((s, target_cls))

    print(f"模式: {args.split_mode} | 随机种子: {args.seed if args.split_mode == 'random' else 'N/A'}")

    # 3. 启动并行渲染
    for split_name, data_list in splits.items():
        if not data_list: continue
        print(f"\n>>> 正在并行处理 {split_name} (共 {len(data_list)} 个样本)...")
        
        tasks = []
        for item, target_cls in data_list:
            tasks.append((
                item["step_path"], 
                item["raw_cls"], 
                target_cls, 
                args.img_root, 
                12, # 固定 12 视角
                TARGET_CLASSES.index(target_cls)
            ))

        final_json = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(render_worker, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Split: {split_name}"):
                result = future.result()
                if result:
                    final_json.append(result)

        # 4. 保存 JSON 标注
        save_path = os.path.join(args.json_dir, f"{split_name}.json")
        with open(save_path, "w") as f:
            json.dump(final_json, f, indent=4)
        print(f"完成！已保存 {len(final_json)} 条记录到 {save_path}")

if __name__ == "__main__":
    main()