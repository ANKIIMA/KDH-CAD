# train.py
import torch
import torch.nn as nn
import json
import swanlab
import warnings
from tqdm import tqdm
import setproctitle
from args import parse_args
from MultiviewDataset import create_dataloaders
from encoderModels.classifierNet import PrototypeShiftClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score
from lightning.pytorch import seed_everything
import gc
import time

@torch.no_grad()
def init_shift_with_data(model, train_loader, args):
    """
    使用训练集图像特征的均值来热启动 shift_vector
    """
    print("\n" + "="*40)
    print(">> [Warm-up] Starting Visual Centroid Calibration...")
    model.eval()
    
    # 使用 float32 累加以保证精度，防止 bfloat16 在大量累加时丢失精度
    dim = model.model_dim
    n_cls = model.n_classes
    sum_features = torch.zeros(n_cls, dim, device=args.device, dtype=torch.float32)
    counts = torch.zeros(n_cls, device=args.device)

    # 1. 遍历训练集收集原始图像特征
    for images_batch, labels in tqdm(train_loader, desc="Calibrating Centroids"):
        # 仅通过 Encoder 提取原始均值特征 [Batch, Dim]
        img_means_raw, _ = model.encoder(images_batch) #
        img_means_raw = img_means_raw.to(torch.float32)
        labels = labels.to(args.device)

        for i in range(labels.size(0)):
            label = labels[i]
            sum_features[label] += img_means_raw[i]
            counts[label] += 1

    # 2. 计算位移并注入参数
    for c in range(n_cls):
        if counts[c] > 0:
            # 视觉聚类中心 (Visual Centroid)
            visual_centroid = (sum_features[c] / counts[c]).to(args.dtype)
            # 教材原型中心 (Text Base)
            text_base = model.base_prototypes[c].to(args.dtype)
            
            # 核心公式：Shift = (Visual - Base) / Scale
            # 确保 forward 时: Base + Shift * Scale = Visual_Centroid
            new_shift = (visual_centroid - text_base) / model.shift_scale
            
            model.shift_vector.data[c] = new_shift
        else:
            print(f"   [Warning] Class {c} ({args.class_names[c] if hasattr(args, 'class_names') else c}) has no data.")

    print(">> [Warm-up] Calibration Complete.")
    print("="*40 + "\n")

def train_model(args):
    setproctitle.setproctitle(f"zyn")
    seed_everything(args.seed, workers=True)

    swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        name=args.exp_name or f"{args.model_type}_{args.classifier_mode}",
        config=vars(args),
        mode=args.swanlab_mode
    )

    # 1. 加载数据
    with open(args.train_json, "r") as f:
        train_data = json.load(f)
    label_to_name = {item["label"]: item["label_name"] for item in train_data}
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]
    
    train_loader, valid_loader, test_loader = create_dataloaders(args)

    # 2. 初始化模型
    model = PrototypeShiftClassifier(args, class_names)
    model.to(args.device)

    if args.use_shift == "yes" and getattr(args, "shift_init_mode", "random") == "data":
        t_shift_start = time.perf_counter()
        init_shift_with_data(model, train_loader, args)
        t_shift_elapsed = time.perf_counter() - t_shift_start
        print(f">> [Timing] Shift warm-up: {t_shift_elapsed:.2f}s")
        swanlab.log({"time/shift_warmup_s": t_shift_elapsed})

    # 3. 设置可训练参数
    trainable_params = []
    for param in model.parameters():
        param.requires_grad = False

    print("\n" + "="*40)
    print(">>> Training Configuration")
    
    print(f">> [Mode] {args.classifier_mode.upper()}")
    for param in model.cross_attn.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    if args.use_shift == "yes":
        model.shift_vector.requires_grad = True
        trainable_params.append(model.shift_vector)
        if hasattr(model, "shift_scale") and isinstance(model.shift_scale, nn.Parameter):
            model.shift_scale.requires_grad = True
            trainable_params.append(model.shift_scale)
        print(">> [Action] Shift Vector: ENABLED")

    if not trainable_params:
        print(">> [Warning] No trainable parameters found!")
        optimizer, scheduler = None, None
    else:
        num_trainable_tensors = len(trainable_params)
        print(f">> Trainable Tensors (Blocks): {num_trainable_tensors}")
        print(f">> Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print("="*40 + "\n")

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    t_train_total_start = time.perf_counter()
    epoch_times = []

    # 5. 训练循环
    for epoch in range(args.epochs):
        t_epoch_start = time.perf_counter()
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for images_batch, labels in pbar:
            logits, projected_embeddings, _, aux_info = model(images_batch)
            labels = labels.to(logits.device)
            
            loss = criterion(logits, labels)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({"Acc": f"{100*correct/total:.1f}%"})

            if aux_info and "sim_base" in aux_info:
                # 获取 Ground Truth 对应的相似度得分
                batch_indices = torch.arange(labels.size(0)).to(labels.device)
                score_base = aux_info["sim_base"][batch_indices, labels].mean().item()
                score_shift = aux_info["sim_shift"][batch_indices, labels].mean().item()
                # print(f"    [Debug] Sim Base: {score_base:.4f}, Sim Shifted: {score_shift:.4f}")

                swanlab.log({
                    "train/sim_base_gt": score_base, 
                    "train/sim_shift_gt": score_shift,
                    "train/shift_scale": model.shift_scale.item() if hasattr(model, "shift_scale") else 1.0
                }, step=epoch)
            
            # --- 显存清理 ---
            del logits, projected_embeddings, loss
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        t_epoch_elapsed = time.perf_counter() - t_epoch_start
        epoch_times.append(t_epoch_elapsed)
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Train Loss {avg_loss:.4f}, Time {t_epoch_elapsed:.1f}s")
        swanlab.log({"train/loss": avg_loss, "train/acc": train_acc, "train/epoch_time_s": t_epoch_elapsed, "epoch": epoch+1}, step=epoch)

        # 验证阶段
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            # 进入验证前强制清理
            torch.cuda.empty_cache()
            gc.collect()
            
            val_acc, val_loss, _, _ = evaluate(model, valid_loader, args.device, criterion)
            print(f"Epoch {epoch+1}: Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}")
            swanlab.log({"valid/acc": val_acc, "valid/loss": val_loss, "epoch": epoch+1}, step=epoch)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.best_model_path)
                print(f"[*] Saved New Best Model: {best_acc:.2f}%")

        if scheduler:
            scheduler.step()

    # 6. 测试阶段
    t_train_total = time.perf_counter() - t_train_total_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n>> [Timing] Total training: {t_train_total:.1f}s, Avg epoch: {avg_epoch_time:.1f}s")
    swanlab.log({"time/train_total_s": t_train_total, "time/avg_epoch_s": avg_epoch_time}, step=args.epochs)

    if args.skip_test == "yes":
        print("[+] Final test skipped (--skip_test yes).")
        swanlab.finish()
        return

    print("\n>>> Final Testing...")
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(args.best_model_path))
    test_acc, test_loss, true_labels, pred_labels = evaluate(model, test_loader, args.device, criterion)

    # Balanced Accuracy
    bacc = balanced_accuracy_score(true_labels, pred_labels) * 100
    # Macro-F1
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    # Micro-F1
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')
    
    print(f"[!] Test Results:")
    print(f"    - Accuracy:  {test_acc:.2f}%")
    print(f"    - BACC:      {bacc:.2f}%")
    print(f"    - Macro-F1:  {macro_f1:.4f}")
    print(f"    - Micro-F1:  {micro_f1:.4f}")
    print(f"    - Test Loss: {test_loss:.4f}")
    print("-" * 30)
    print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))
    
    swanlab.log({
        "test/acc": test_acc,
        "test/bacc": bacc,
        "test/macro_f1": macro_f1,
        "test/micro_f1": micro_f1,
        "test/loss": test_loss,
        "time/train_total_s": t_train_total,
        "time/avg_epoch_s": avg_epoch_time
    }, step=args.epochs + 1)
    swanlab.finish()

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images_batch, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            logits, _, _, _ = model(images_batch)
            labels = labels.to(logits.device)
            total_loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 及时释放局部变量
            del logits
    return accuracy_score(all_labels, all_preds)*100, total_loss/len(dataloader), all_labels, all_preds

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_args()
    train_model(args)
