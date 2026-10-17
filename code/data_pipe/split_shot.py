import json
import os
import argparse
from collections import defaultdict


def rebuild_full_ordered(json_base_dir, ref_shot=50):
    ref_dir = os.path.join(json_base_dir, f"shot_{ref_shot}")
    with open(os.path.join(ref_dir, "train.json"), "r") as f:
        ref_train = json.load(f)
    with open(os.path.join(ref_dir, "valid.json"), "r") as f:
        ref_valid = json.load(f)
    with open(os.path.join(ref_dir, "test.json"), "r") as f:
        ref_test = json.load(f)

    by_label = defaultdict(list)
    for item in ref_train:
        by_label[item["label"]].append(item)
    for item in ref_valid:
        by_label[item["label"]].append(item)
    for item in ref_test:
        by_label[item["label"]].append(item)

    return by_label


def split_shot(by_label, shot, json_base_dir):
    valid_per_class = 50

    new_train = []
    new_valid = []
    new_test = []
    for label in sorted(by_label.keys()):
        items = by_label[label]
        new_train.extend(items[:shot])
        new_valid.extend(items[shot : shot + valid_per_class])
        new_test.extend(items[shot + valid_per_class :])

    save_dir = os.path.join(json_base_dir, f"shot_{shot}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "train.json"), "w") as f:
        json.dump(new_train, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "valid.json"), "w") as f:
        json.dump(new_valid, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "test.json"), "w") as f:
        json.dump(new_test, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] shot_{shot}: train={len(new_train)}, valid={len(new_valid)}, test={len(new_test)}, "
        f"total={len(new_train) + len(new_valid) + len(new_test)}"
    )

    train_keys = set(x["part_key"] for x in new_train)
    valid_keys = set(x["part_key"] for x in new_valid)
    test_keys = set(x["part_key"] for x in new_test)
    assert len(train_keys & valid_keys) == 0, "train-valid leakage!"
    assert len(train_keys & test_keys) == 0, "train-test leakage!"
    assert len(valid_keys & test_keys) == 0, "valid-test leakage!"
    print(f"    No leakage verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_base_dir", type=str, required=True)
    parser.add_argument("--shots", type=int, nargs="+", required=True)
    parser.add_argument("--ref_shot", type=int, default=50,
                        help="Reference shot to rebuild full ordered data from. "
                             "Use 50 for shots <= 50, 100 for shots 51-100.")
    args = parser.parse_args()

    by_label = rebuild_full_ordered(args.json_base_dir, args.ref_shot)

    for shot in args.shots:
        max_train = len(next(iter(by_label.values())))
        if shot + 50 > max_train:
            print(f"[SKIP] shot_{shot}: need {shot + 50}/class but only {max_train} available from ref_shot={args.ref_shot}")
            continue
        split_shot(by_label, shot, args.json_base_dir)
