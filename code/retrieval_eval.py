import torch
import torch.nn.functional as F
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from encoderModels.classifierNet import PrototypeShiftClassifier
from MultiviewDataset import STPDataset, custom_collate_fn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    for images_batch, labels in tqdm(dataloader, desc="Extracting embeddings"):
        # The classifier's second return value is the representation immediately
        # before its final similarity/classification step.
        _, embeddings, _, _ = model(images_batch)

        all_embeddings.append(embeddings.detach().cpu().float())
        all_labels.extend(labels.numpy().tolist())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings, all_labels


def compute_retrieval_metrics(query_embs, query_labels, gallery_embs, gallery_labels, ks=(1, 5, 10)):
    query_embs = F.normalize(query_embs, p=2, dim=1)
    gallery_embs = F.normalize(gallery_embs, p=2, dim=1)

    sim_matrix = torch.mm(query_embs, gallery_embs.t())

    num_queries = sim_matrix.shape[0]
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    recalls = {f"Recall@{k}": 0.0 for k in ks}
    aps = []

    for i in range(num_queries):
        query_label = query_labels[i]
        ranked_labels = [gallery_labels[idx] for idx in sorted_indices[i].tolist()]

        for k in ks:
            if query_label in ranked_labels[:k]:
                recalls[f"Recall@{k}"] += 1.0

        relevant_positions = [j + 1 for j, lbl in enumerate(ranked_labels) if lbl == query_label]
        if len(relevant_positions) > 0:
            cumsum = np.cumsum([1.0 if ranked_labels[j] == query_label else 0.0 for j in range(len(ranked_labels))])
            precisions = [cumsum[j] / (j + 1) for j in range(len(ranked_labels)) if ranked_labels[j] == query_label]
            aps.append(np.mean(precisions))
        else:
            aps.append(0.0)

    for k in ks:
        recalls[f"Recall@{k}"] /= num_queries

    mAP = np.mean(aps)
    return recalls, mAP


def main():
    import sys
    from args import parse_args

    override = argparse.ArgumentParser()
    override.add_argument("--model_path", type=str, required=True)
    override.add_argument("--gallery_json", type=str, required=True)
    override.add_argument("--query_json", type=str, required=True)
    override.add_argument("--batch_size", type=int, default=8)
    cli_args, remaining = override.parse_known_args()

    sys.argv = [
        sys.argv[0], *remaining,
        "--train_json", cli_args.gallery_json,
        "--valid_json", cli_args.query_json,
        "--test_json", cli_args.query_json,
    ]
    args = parse_args()

    args.model_path = cli_args.model_path
    args.gallery_json = cli_args.gallery_json
    args.query_json = cli_args.query_json
    if cli_args.batch_size:
        args.batch_size = cli_args.batch_size

    with open(args.gallery_json, "r") as f:
        gallery_data = json.load(f)
    label_to_name = {item["label"]: item["label_name"] for item in gallery_data}
    class_names = [label_to_name[i] for i in sorted(label_to_name.keys())]

    model = PrototypeShiftClassifier(args, class_names).to(args.device)
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] Model loaded from {args.model_path}")

    gallery_dataset = STPDataset(args.gallery_json, num_views=12)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    query_dataset = STPDataset(args.query_json, num_views=12)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)

    print(">>> Extracting gallery embeddings...")
    gallery_embs, gallery_labels = extract_embeddings(model, gallery_loader, args.device)
    print(f"Gallery: {gallery_embs.shape[0]} samples")

    print(">>> Extracting query embeddings...")
    query_embs, query_labels = extract_embeddings(model, query_loader, args.device)
    print(f"Query: {query_embs.shape[0]} samples")

    print(">>> Computing retrieval metrics...")
    recalls, mAP = compute_retrieval_metrics(query_embs, query_labels, gallery_embs, gallery_labels)

    print("\n" + "=" * 40)
    print("Retrieval Results")
    print("=" * 40)
    for k, v in recalls.items():
        print(f"  {k}: {v*100:.2f}%")
    print(f"  mAP: {mAP*100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
