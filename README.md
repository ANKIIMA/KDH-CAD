# KDH-CAD: Knowledge–data hybrid CAD learning under data scarcity—A case study on part classification

- By Ziqin Gao, Zhijie Yang, and Qiang Zou
- Email: [qzou.code@gmail.com](mailto:qzou.code@gmail.com)
- Webpage: https://qiang-zou.github.io/
- Latest Release: 2026.09

![General KDH-CAD architecture](figures/kdhcad-general-architecture.png)

## !important

The source code was developed as a byproduct of the projects and methods presented in KDH-CAD (see reference), which is a knowledge--data hybrid learning method for CAD under data scarcity. It keeps a pre-trained vision-language model frozen, uses domain knowledge from textbooks and tutorials to complete CAD concepts, and calibrates them with a small amount of labeled CAD data through Shift Vector and cross-attention for part classification and model retrieval.

It can be run with Pytorch 2.6.0 + CUDA 12.1 on the operating system Ubuntu, Windows and Mac.

## 1. Copyright

- KDH-CAD is GNU licensed. It is developed and maintained by Ziqin Gao, Zhijie Yang, and Qiang Zou for research use. All rights of the program are reserved by Qiang Zou. This Python source code is available only for primary use for academic purposes. No secondary use, such as copy, distribution, diversion, business purpose, etc., is allowed. In no event shall the author be liable to any party for direct, indirect, special, incidental, or consequential damage arising out of the use of this program. KDH-CAD is self-contained.

## 2. Download

- Source code: this repository.
- TMCAD: obtain it from the [BRT dataset documentation](https://github.com/Qiang-Zou/BRT/blob/main/DATASET.md).
- FabWave: obtain it from the [official FabWave download page](https://www.dimelab.org/fabwave-dataset-downloads).
- Qwen3-VL-2B: obtain it from [ModelScope: Qwen3-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct).

You can directly download the best checkpoints trained with the 1k-sample setting used in the paper:

```bash
cp configs/download.env.example configs/download.env
# Edit configs/download.env and add the checkpoint direct links
bash scripts/download_checkpoints.sh
```

This script downloads checkpoints only. It supports resumable direct downloads. You can also download the checkpoint manually and set its relative path in `configs/paths.env`.

| Purpose | Configuration key | Release link |
|---|---|---|
| Best checkpoint trained on 1k samples of TMCAD | `KDH_CAD_TMCAD_CKPT` | https://pan.baidu.com/s/1x2C3lgjOcZzu_GzfoMgbLw?pwd=53ky |

Relative setting is included in `configs/paths.env.example`.

## 3. Repository Structure

```text
kdhcad-release/
├── code/                 # Paper core: training, testing, retrieval, and knowledge encoding
│   └── data_pipe/        # STEP-to-multi-view conversion and few-shot split tools
├── configs/              # Local-path and checkpoint-download templates
├── figures/              # General KDH-CAD architecture figure
├── scripts/              # Model/checkpoint downloads and two training/test entry points
└── environment.yaml
```

## 4. Installing (Linux + Pytorch 2.6.0 + CUDA 12.1)

```bash
conda env create -f environment.yaml
conda activate kdhcad
cp configs/paths.env.example configs/paths.env
```

Edit `configs/paths.env` and provide the paths to the prepared multi-view data, data splits, knowledge embeddings, Qwen3-VL-2B, and checkpoints. All paths in the template are relative to the release directory; use relative paths such as `../kdhcad-assets` to store large files outside the repository.

Download the pre-trained VLM model:

```bash
pip install -U modelscope
bash scripts/download_qwen3vl2b.sh
```

This script is equivalent to:

```bash
modelscope download --model Qwen/Qwen3-VL-2B-Instruct \
  --local_dir "${KDH_CAD_QWEN_PATH}"
```

## 5. Usage

The training scripts use only train/validation data to select the best weights and do not evaluate on the final test set. Replace the corresponding `*_CKPT` path in `configs/paths.env` with the training output, then run the test script.

### 5.1 Multi-view Data Preparation

KDH-CAD consumes 12 rendered views per CAD model. The scripts in `code/data_pipe/` convert raw STEP files into multi-view images and the train/valid/test split JSONs used by `code/MultiviewDataset.py`. If you already use the prepared data splits (see Section 2), you can skip this step.

These scripts require `pythonocc-core` (not included in `environment.yaml`), plus `tqdm` and `loguru`:

```bash
conda install -c conda-forge pythonocc-core
pip install loguru
```

Rendering needs an OpenGL display. On a headless server, wrap the commands with `xvfb-run -a` as shown below.

TMCAD (paper main setting: 1,000 train / 500 valid / remaining test; the bolt and screw directories are merged into one class):

```bash
xvfb-run -a python code/data_pipe/step_to_multiview_tmcad.py \
    --src_root /path/to/TMCAD \
    --img_root /path/to/tmcad_multiviews \
    --json_dir /path/to/splits/shot_100 \
    --n_train 100 --n_val 50 --num_workers 8
```

`--n_train` sets the per-class training size (100 gives the shot_100 main setting above). Smaller few-shot splits can also be derived from a larger split with `code/data_pipe/split_shot.py`. The default `--split_mode random --seed 42` reproduces the paper splits on the original filesystem; because the shuffle depends on the directory listing order, use the provided split JSONs to reproduce the paper numbers exactly, or use `--split_mode sequential` for a deterministic index-based split.

FabWave (only the categories with at least 200 models are used; validation/test are the fixed sorted tail with 50 models per class each):

```bash
xvfb-run -a python code/data_pipe/step_to_multiview_fabwave.py \
    --src_root /path/to/FebWave_Filtered \
    --img_root /path/to/fabwave_multiviews \
    --json_dir /path/to/fabwave_splits/tail_50 \
    --n_train 50 --num_workers 8
```

The FabWave script keeps the eight labeled categories as provided by the dataset (Keyway_Shaft and Rotary_Shaft stay separate); merge them into a single Shafts category as described in the paper if needed. Both scripts skip already-rendered images, so interrupted runs can be resumed with the same arguments. Point `KDH_CAD_TMCAD_SPLITS` / `KDH_CAD_FABWAVE_SPLIT_DIR` in `configs/paths.env` at the resulting JSON directories.

### 5.2 Model Classification and Retrieval

Train the model:

```bash
bash scripts/train_main_results.sh
```

Test checkpoint or your own trained weights on classification or model retrieval:

```bash
bash scripts/test_main_results.sh
```

This test entry point reports Accuracy, BACC, Macro-F1, Micro-F1, and a classification report for TMCAD and FabWave. For TMCAD, it also reports Recall@1, Recall@5, Recall@10, and mAP for model retrieval.

## 6. References

If you use this code, please cite:

```bibtex
@article{gao2027kdhcad,
  title   = {KDH-CAD: Knowledge--data hybrid CAD learning under data scarcity---A case study on part classification},
  author  = {Gao, Ziqin and Yang, Zhijie and Zou, Qiang},
  journal = {Computer-Aided Design},
  volume  = {202},
  pages   = {104158},
  year    = {2027},
  doi     = {10.1016/j.cad.2026.104158},
  url     = {https://doi.org/10.1016/j.cad.2026.104158}
}
```

Please also cite and follow the license terms of TMCAD, FabWave, and Qwen3-VL-2B when using those external resources.
