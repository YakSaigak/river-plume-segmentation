# River Plume Segmentation — Training and Evaluation Code

Reproducibility package for the paper on automatic segmentation of river
plumes in Sentinel-2 and Landsat imagery with an ensemble of deep learning
models.

The repository contains one Jupyter notebook per model (12 models), a data
preparation notebook, an ensemble notebook, a stacking baseline, a
post-processing study and a statistical-analysis notebook.

## Repository structure

```
notebooks/
  00_data_preparation.ipynb        # augmented HDF5 datasets for the Keras models
  01_unet_vgg16.ipynb              # U-Net + VGG16
  02_unet_resnet50.ipynb           # U-Net + ResNet50
  03_unet_xception.ipynb           # U-Net + Xception
  04_deeplabv3plus_vgg16.ipynb     # DeepLabv3+ + VGG16
  05_deeplabv3plus_resnet50.ipynb  # DeepLabv3+ + ResNet50
  06_deeplabv3plus_xception.ipynb  # DeepLabv3+ + Xception (same ASPP decoder as 04/05)
  07_fcn_vgg16.ipynb               # FCN + VGG16
  08_fcn_resnet50.ipynb            # FCN + ResNet50
  09_fcn_xception.ipynb            # FCN-8s + Xception
  10_yolo11seg.ipynb               # YOLO11l-seg (Ultralytics)
  11_segformer.ipynb               # SegFormer-B3 (HuggingFace)
  12_mask2former.ipynb             # Mask2Former Swin-S (HuggingFace)
  13_ensemble.ipynb                # weighted averaging of the four best models
  14_stacking_random_forest.ipynb  # stacking ensemble with a Random Forest meta-model
  15_postprocessing_methods.ipynb  # post-processing experiments on ensemble predictions
  16_statistical_analysis.ipynb    # per-scene metrics CSV, bootstrap CIs, Wilcoxon tests, area-fidelity metrics
requirements.txt
README.md
```

## Data availability

The dataset (Sentinel-2 and Landsat RGB composites and manually annotated
binary plume masks, split into `train` / `val` / `test`) is **not**
redistributed in this repository. The imagery is freely available from the
public archives, and the contours plus the exact scene list are available
through the **See The Sea (STS)** information system of the Space Research
Institute of the Russian Academy of Sciences (IKI RAS):

- Sentinel-2 MSI — Copernicus Data Space Ecosystem
  (<https://dataspace.copernicus.eu>).
- Landsat-8/9 OLI/OLI-2 — USGS EarthExplorer
  (<https://earthexplorer.usgs.gov>).
- Expert plume contours and the scene list — the STS service of IKI RAS,
  available on free registration arranged through the IKI RAS administration.

Trained model weights are not distributed. Every notebook trains its model
from publicly available pretrained checkpoints (ImageNet for the CNN encoders,
COCO for YOLO11l-seg, ADE20K for SegFormer and Mask2Former), so all results
can be reproduced from the code and the dataset alone.

Expected directory layout used by every notebook:

```
split_data/
  train/{images, Masks}
  val/{images, Masks}
  test/{images, Masks}
```


## Hardware and environment

All models were trained in a local Jupyter environment on a single NVIDIA
GeForce RTX 3050 GPU (8 GB VRAM). Per-model batch sizes and encoder freezing
depths were chosen to fit this memory budget (see the configuration cell of
each notebook and the table below).

Python >= 3.10 with the packages from `requirements.txt`. The Keras notebooks
set `TF_USE_LEGACY_KERAS=1` and require the `tf-keras` package on
TensorFlow >= 2.16.

```
pip install -r requirements.txt
```

Reference environment used for the reported runs: Python 3.10, CUDA 12.1,
cuDNN 8.9, TensorFlow 2.16 + tf-keras 2.16, PyTorch 2.2, transformers 4.40,
ultralytics 8.3. Pin these (or use the exact versions in `requirements.txt`)
for the closest reproduction.

A fixed random seed (42) is set in every notebook. **Owing to GPU-level
nondeterminism (cuDNN kernel selection, atomic operations, library and
driver versions), re-training reproduces the reported metrics statistically
rather than bit-exactly.** Do not expect bit-identical numbers on a re-run.

## Run order

1. `00_data_preparation.ipynb` — produces the train/val HDF5 files for the
   three encoder normalizations. Augmentation is applied to the **training
   split only**: each training scene is written once as the original plus five
   augmented variants (six samples per scene), giving 3,090 training samples.
   Masks are resampled with nearest-neighbour interpolation.
2. `01`–`09` — train the nine Keras models. Each notebook ends with the
   test-set evaluation and the export of probability/binary prediction masks.
   Notebooks `02` and `05` additionally export the validation predictions
   (`val_masks/...`, `probability_masks/val/...`) used by the ensemble and
   stacking notebooks.
3. `10_yolo11seg.ipynb` — requires the dataset converted to the YOLO
   segmentation format (`data.yaml` with polygon labels). Also exports the
   validation predictions used by the ensemble and stacking notebooks.
4. `11_segformer.ipynb` and `12_mask2former.ipynb` — train directly on the
   image/mask folders. Notebook `12` also exports the validation predictions
   used by the ensemble and stacking notebooks.
5. `13_ensemble.ipynb` — computes the validation Dice of the four base models
   from their exported validation predictions (one identical per-scene
   definition for all four), selects the averaging scheme on the validation
   split, and evaluates the selected configuration once on the test split.
   Test scores of the non-selected schemes are printed for reference only.
   Requires the validation and test exports of notebooks `02`, `05`, `10`
   and `12`; no weights are hard-coded.
6. `16_statistical_analysis.ipynb` — recomputes per-scene Dice for all models
   and the ensemble, writes `results/per_scene_dice.csv`, and produces the
   bootstrap confidence intervals, the paired Wilcoxon tests and the
   area-fidelity metrics. Run this after 01–13 have exported their test predictions.

## Training protocol

All nine Keras models share one optimization protocol: input 256×256×3,
encoder-specific Keras `preprocess_input` normalization, loss = 0.6·Dice +
0.4·Focal(α = 0.25, γ = 2.0), Adam with gradient-norm clipping 0.5, learning
rate selected per model with a Learning Rate Finder sweep (1e-6…1e-2,
50 single-batch steps; the LR at the loss minimum divided by 10), up to
300 epochs with `EarlyStopping` (patience 30, monitoring validation Dice,
restoring the best weights), `ReduceLROnPlateau` (factor 0.7, patience 8,
min LR 1e-7) and best-checkpoint saving on the same metric. Prediction
threshold 0.5.

Light dropout is used in the CNN decoders (per-model values in the table
below): 0.05 in the U-Net/FCN decoder blocks and the DeepLabv3+ decoder,
0.1 in the U-Net bottleneck and the ASPP module, 0.02 in the final U-Net
refinement blocks. The Xception-based U-Net and FCN variants (notebooks
03 and 09) use no decoder dropout.

All models except YOLO11-seg use the same augmentation policy (horizontal and
vertical flips with p = 0.5, rotation within ±25° with p = 0.9, and adaptive
brightness-dependent photometric transforms): the Keras models via the
augmented HDF5 training files, SegFormer and Mask2Former on-the-fly during
training. YOLO11-seg uses the native Ultralytics augmentation. Geometric
transforms use bilinear interpolation for images and **nearest-neighbour for
masks** to preserve discrete labels.

| # | Model | Batch | Encoder freezing | Decoder dropout |
|---|-------|-------|------------------|-----------------|
| 01 | U-Net + VGG16 | 8 | first 10 of 19 layers | 0.05 dec. / 0.1 bottleneck / 0.02 final |
| 02 | U-Net + ResNet50 | 4 | first 100 of 175 layers | 0.05 dec. / 0.1 bottleneck / 0.02 final |
| 03 | U-Net + Xception | 16 | first 100 of 132 layers | none |
| 04 | DeepLabv3+ + VGG16 | 4 | first 10 of 19 layers | 0.1 ASPP / 0.05 dec. |
| 05 | DeepLabv3+ + ResNet50 | 4 | first 100 of 175 layers | 0.1 ASPP / 0.05 dec. |
| 06 | DeepLabv3+ + Xception | 2 | first 100 of 132 layers | 0.1 ASPP / 0.05 dec. |
| 07 | FCN + VGG16 | 8 | first 10 of 19 layers | 0.05 dec. (final block none) |
| 08 | FCN + ResNet50 | 4 | first 100 of 175 layers | 0.05 dec. (final block none) |
| 09 | FCN-8s + Xception | 16 | first 100 of 132 layers | none |

Layer counts refer to the Keras `include_top=False` models; note that the
frozen fraction is therefore larger for Xception (100/132 ≈ 76%) than for
VGG16 (10/19 ≈ 53%) and ResNet50 (100/175 ≈ 57%).

Transformer and detection models:

| # | Model | Input | Batch | Epochs (patience) | Optimizer / LR | Schedule | Notes |
|---|-------|-------|-------|-------------------|----------------|----------|-------|
| 10 | YOLO11l-seg (COCO pretrained) | 640 | 8 | 200 (30) | AdamW, lr0 = 1e-4, wd = 0.01, momentum = 0.9 | cosine | no layers frozen (freeze = 0); dropout 0.1, degrees 180, warmup 3, AMP; optimizer settings selected by a two-stage grid search on the validation split |
| 11 | SegFormer-B3 (ADE-512 pretrained) | 512 | 8 | 100 (8, val loss) | AdamW, head 1e-4 / encoder 1e-5, wd = 0.01, grad clip 1.0 | CosineAnnealingLR | first 70% of encoder parameter tensors frozen |
| 12 | Mask2Former Swin-S (ADE pretrained) | 640 | 2 | 100 (15) | AdamW, 5e-5, wd = 0.01, grad clip 1.0 | CosineAnnealingWarmRestarts (T₀ = 10, T_mult = 2, η_min = 5e-7) | Swin backbone frozen|

Ensemble (`13_ensemble.ipynb`): weighted averaging of the probability or
binary maps of YOLO11-seg, DeepLabv3+ ResNet50, Mask2Former and
U-Net ResNet50. Three schemes are compared: equal weights, weights
proportional to the validation Dice of each model, and a top-2 scheme (the
two models with the highest validation Dice, Dice-proportional weights). The
comparison of all schemes and the choice of the working configuration are
performed on the validation split only; the selected scheme is then evaluated
once on the test split. The validation Dice values that define the weights are
computed inside the notebook from the exported validation predictions of the
four models (per-scene Dice against the validation ground truth, the same
definition for each model); no weights are hard-coded in the repository. The
configuration adopted in the paper is the Dice-weighted averaging of the
probability maps.

## Evaluation protocol

All models are evaluated with the same metric set on the test split: Dice,
Accuracy, Precision, Recall and Hausdorff distances (HD95, Average HD,
Max HD) computed symmetrically with Euclidean distance transforms over the
full binary masks of each image, at the native resolution of the ground-truth
masks (predictions are upsampled with nearest-neighbour resampling before
comparison). Hausdorff distances are reported in pixels at the native
ground-truth resolution.
