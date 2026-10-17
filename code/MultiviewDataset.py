# MultiviewDataset.py
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random

class STPDataset(Dataset):
    def __init__(self, data_path, num_views=12):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.num_views = num_views
        
        # 简单的路径检查
        valid_data = []
        for item in self.data:
            if len(item.get("image_paths", [])) >= self.num_views:
                 valid_data.append(item)
        self.data = valid_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = item["image_paths"]
        label = item["label"]
        
        images = []
        # 确保只读取前 num_views 张图片
        for path in image_paths[:self.num_views]:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                #由于是训练，如果坏图可以补一张全黑或随机图防止崩坏，或者报错
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        return images, label 

def custom_collate_fn(batch):
    """
    自定义collate_fn处理 PIL Image 列表
    Batch 结构:
    - images: List[List[PIL.Image]] (batch_size, num_views)
    - labels: Tensor (batch_size)
    """
    images_batch = [item[0] for item in batch]
    labels_batch = torch.tensor([item[1] for item in batch])
    return images_batch, labels_batch

def create_dataloaders(args):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    # 创建数据集
    train_dataset = STPDataset(args.train_json)
    valid_dataset = STPDataset(args.valid_json)
    test_dataset = STPDataset(args.test_json)
    
    # 创建DataLoader，使用自定义 collate_fn
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, 
                             collate_fn=custom_collate_fn, drop_last=True,
                             worker_init_fn=seed_worker, generator=g)
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, 
                             collate_fn=custom_collate_fn, drop_last=False,
                             worker_init_fn=seed_worker, generator=g)
                             
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, 
                             collate_fn=custom_collate_fn, drop_last=False,
                             worker_init_fn=seed_worker, generator=g)
    
    return train_loader, valid_loader, test_loader