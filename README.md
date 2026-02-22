# Off-Road Semantic Segmentation for Tracked Robot

Real-time semantic segmentation system for a caterpillar-track autonomous robot operating on unpaved terrain. Trained on three open-source off-road datasets and deployed to a Qualcomm edge device for onboard traversability estimation.

```
Camera (1280×1080) → DDRNet23-Slim (INT8) → 7-class segmentation → Traversability map
                     Qualcomm IQ-9075 NPU
                     ~15ms/frame (60+ FPS)
```

---

## Key Design Decisions

### Why DDRNet23-Slim over EfficientViT-B1?

EfficientViT-B1 was originally selected for its higher accuracy (80.5% mIoU on Cityscapes). However, when converting to INT8 for the Qualcomm Hexagon NPU, the **Smooth Ground class collapsed to 0%** — the most critical class for path planning.

The root cause is architectural: Vision Transformers rely on Softmax, LayerNorm, and GELU — all of which produce distributions that INT8's 256 discrete levels cannot faithfully represent. CNN operations (Conv + BatchNorm + ReLU) are inherently INT8-safe.

| | EfficientViT-B1 | **DDRNet23-Slim** |
|---|---|---|
| Architecture | Vision Transformer | CNN (Dual-Resolution) |
| Parameters | 4.8M | 5.7M |
| Cityscapes mIoU | 80.5% | 77.8% |
| INT8 Quantization | ❌ Smooth Ground collapse | ✅ All ops safe |
| Qualcomm NPU | Softmax/LN/GELU unsupported | 131/131 ops on NPU |
| AI Hub Verified | No | Yes |

### Why Focal Loss with Differential Learning Rate?

Two bugs caused **mode collapse** (all pixels predicted as Obstacle) in initial training:

**Bug 1 — Focal Loss `p_t` distortion:** Passing `weight=alpha` into `F.cross_entropy` corrupts the focal modulation term. Instead of `p_t`, you get `p_t^alpha`, which makes low-alpha classes (Sky, Vegetation) appear "already well-classified" even when they're not. Alpha must be applied **outside** the CE computation.

**Bug 2 — Uniform learning rate destroys pretrained features:** With a single LR=1e-3 for all parameters, the Cityscapes-pretrained backbone features are destroyed within 2 epochs — before the new head learns anything useful. The fix: backbone at 1/10th the head's learning rate.

| Fix | Before | After |
|-----|--------|-------|
| Focal Loss alpha | Inside `F.cross_entropy(weight=)` | Separate multiplication after CE |
| Min class weight | Sky=0.3, Veg=0.5 | All ≥ 1.0 |
| Backbone LR | 1e-3 (same as head) | 1e-4 (10× lower) |
| Warmup | 20 epochs | 5 epochs |

**Result:** mIoU went from 4.6% (1 class) to 22.9% (6/7 classes) at epoch 10.

---

## 7-Class Ontology

Designed for caterpillar-track robot traversability:

| ID | Class | Description | Action |
|----|-------|-------------|--------|
| 0 | Smooth Ground | Asphalt, concrete, packed dirt | ✅ Optimal path |
| 1 | Rough Ground | Sand, gravel, mud, snow | ⚠️ Slow down |
| 2 | Vegetation | Low grass, moss, leaves | ⚠️ Passable (tracks) |
| 3 | Obstacle | Trees, rocks, buildings, fences, bushes | ❌ Avoid |
| 4 | Water | Puddles, streams, lakes | ❌ Avoid (flood risk) |
| 5 | Sky | Sky, clouds | — Ignore |
| 6 | Dynamic | People, vehicles, animals | ❌ Avoid (safety) |

**Caterpillar-specific mappings:** bush → Obstacle (track entanglement), puddle → Water (drivetrain flooding), dirt → Smooth Ground (ideal surface for tracks).

---

## Datasets

Three open-source datasets merged into the unified 7-class ontology. No target camera data is used during training.

| Dataset | Train Images | Environment | Original Classes | Link |
|---------|-------------|-------------|-----------------|------|
| **RELLIS-3D** | 4,169 | Military test trails | 20 → 7 | [GitHub](https://github.com/unmannedlab/RELLIS-3D) |
| **RUGD** | 7,436 | Parks, trails, forests | 24 → 7 | [rugd.vision](http://rugd.vision/) |
| **GOOSE** | 7,845 | European outdoor/forest | 64 → 7 | [goose-dataset.de](https://goose-dataset.de/) |
| **Total** | **19,450** | | | |

Validation: 1,788 images (RELLIS-3D 30% test split).

---

## Training

### Quick Start

```bash
# Setup
bash setup.sh && conda activate offroad

# Download datasets into data/ (see data/README.md)
python scripts/make_split_custom.py
python scripts/preprocess_datasets.py   # ~2 min, one-time

# Train
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8

# Quiet mode (one line per epoch)
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8 --quiet
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | DDRNet23-Slim (5.7M) | INT8-safe, Qualcomm AI Hub verified |
| Pretrained | Cityscapes 19-class | Transfer low-level features (edges, textures) |
| Input size | 544×640 (H×W) | ~0.5× of S10 Ultra (1080×1280), preserves aspect ratio |
| Batch size | 8 | 24GB VRAM optimal; ~2.8M pixels/batch supervision |
| Epochs | 200 | |
| Optimizer | AdamW (wd=0.01) | Decoupled weight decay, standard for fine-tuning |
| **Backbone LR** | **1e-4** | Preserve pretrained features, slow adaptation |
| **Head LR** | **1e-3** | Fast learning for new 7-class classification |
| LR schedule | 5-epoch warmup → 195-epoch cosine | Warmup stabilizes Xavier-init head; cosine for smooth decay |
| Loss | Focal Loss (γ=2.0) | Auto-suppresses easy/large-area classes via `(1-p_t)²` |
| EMA | decay=0.9999 | ~2.9 epoch half-life; smooths batch noise (+0.5–2% mIoU) |
| AMP | FP16 | ~1.5× speed on Tensor Cores; FP32 master weights |
| Grad clip | max_norm=5.0 | Safety net for AMP + class-imbalanced batches |

### Focal Loss Class Weights (Alpha)

| Class | Weight | Rationale |
|-------|--------|-----------|
| Smooth Ground | 1.5 | Core path — moderate emphasis |
| Rough Ground | 3.0 | Rare but affects speed decisions |
| Vegetation | 1.0 | Baseline — gamma handles area suppression |
| Obstacle | 1.0 | Baseline — abundant and diverse |
| Water | 5.0 | Very rare, safety-critical |
| Sky | 1.0 | Baseline — gamma handles area suppression |
| Dynamic | 5.0 | Very rare, safety-critical |

**Design rule:** minimum alpha is 1.0. Gamma (=2.0) auto-suppresses large-area easy classes. Setting alpha below 1.0 starves gradient in early training and causes mode collapse.

### Differential Learning Rate

The backbone (Cityscapes-pretrained) and head (Xavier-initialized) require different learning rates:

```
Backbone (161 tensors):  LR = 1e-4  ← preserve edge/texture features
Head (7 tensors):        LR = 1e-3  ← learn 7-class mapping quickly
```

Without this, the pretrained backbone is destroyed within 2 epochs (loss drops 89% in epoch 1→2 — not learning, but feature collapse). The head then has nothing useful to classify, defaulting to the majority class.

```bash
# Default: 10× ratio (recommended)
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8

# Custom ratio
python scripts/train.py --bb_lr_factor 0.05 ...  # 20× ratio
```

### Augmentation Pipeline

Training: `RandomScale(0.5–2.0) → Pad → RandomCrop(544×640) → HFlip → PhotometricAug → Normalize`

Validation: `Resize(544×640) → Normalize`

| Augmentation | Parameters | Purpose |
|-------------|-----------|---------|
| ColorJitter | brightness/contrast/saturation=0.4, hue=0.15 | Camera domain gap compensation |
| Gaussian Blur | p=0.3, radius=0.5–2.0 | Motion blur / defocus simulation |
| Random Shadow | p=0.2, darkening=0.3–0.7 | Tree canopy shadow simulation |
| Random Erasing | p=0.3, scale=0.02–0.2 | Occlusion robustness |
| Random Grayscale | p=0.05 | Faded lighting conditions |

### Fast Mode

Pre-resizes images from large PNGs (~5MB) to max-1024px JPEGs (~100KB), cutting epoch time from ~470s to ~180s with no quality loss.

```bash
python scripts/preprocess_datasets.py   # one-time, ~2 min
python scripts/train.py --fast ...
```

---

## Deployment (IQ-9075)

```
PyTorch (.pth)  →  ONNX (.onnx)  →  QNN Context Binary (.bin)  →  IQ-9075 NPU
    FP32              opset 17           INT8 quantized              ~15ms/frame
```

### Export

```bash
# ONNX only (for testing on host PC)
python scripts/export_qnn.py \
    --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
    --method onnx

# Full deployment via Qualcomm AI Hub (recommended)
python scripts/export_qnn.py \
    --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
    --method hub
```

### Inference on IQ-9075

```bash
# Video file
python scripts/infer_qnn_video.py \
    --model deploy/ddrnet23_slim_int8.bin \
    --input video.mp4 --output result.mp4

# ROS2 live camera
python scripts/infer_qnn_ros2.py \
    --model deploy/ddrnet23_slim_int8.bin \
    --topic /camera/s10_ultra/color/image_raw
```

ROS2 published topics:

| Topic | Type | Content |
|-------|------|---------|
| `~/segmentation` | sensor_msgs/Image (mono8) | 7-class mask (0–6) |
| `~/costmap` | nav_msgs/OccupancyGrid | Navigation costmap |
| `~/overlay` | sensor_msgs/Image (bgr8) | Blended visualization |

### Preprocessing Requirement

The model does **not** include normalization. Raw pixels (0–255) must be preprocessed before inference:

```python
# Required for ALL inference paths
img = img.astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet mean/std
```

This is standard practice for Qualcomm NPU deployment — preprocessing is more efficient on CPU while NPU handles Conv/BN/ReLU.

---

## Training Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 4000 Blackwell (24GB GDDR7) |
| CPU | AMD Ryzen 9700X |
| RAM | 64GB DDR5 |
| Framework | PyTorch 2.10.0, CUDA 12.8 |
| Camera | MRDVS S10 Ultra RGB-D (1280×1080) |
| Deploy target | Qualcomm Dragonwing IQ-9075 (100 TOPS NPU) |

---

## Project Structure

```
VIL-TerrainSegmentation/
├── scripts/
│   ├── train.py                   # Training (differential LR, Focal Loss)
│   ├── evaluate.py                # mIoU evaluation
│   ├── infer_cam.py               # PyTorch inference (image/video)
│   ├── infer_qnn_video.py         # IQ-9075 video inference (ONNX/QNN)
│   ├── infer_qnn_ros2.py          # IQ-9075 ROS2 live inference
│   ├── export_onnx.py             # Generic ONNX export
│   ├── export_qnn.py              # QNN export for IQ-9075
│   ├── preprocess_datasets.py     # Fast mode preprocessing
│   ├── make_split_custom.py       # 70/30 split creation
│   ├── visualize_predictions.py   # Prediction visualization
│   └── verify_all.py              # Environment verification
├── src/
│   ├── dataset.py                 # Datasets, class mappings, Focal Loss, EMA
│   ├── models.py                  # Model factory (dispatches to DDRNet/ViT/FFNet)
│   └── models_ddrnet.py           # DDRNet23-Slim builder (qai_hub_models)
├── data/                          # Datasets (download separately)
│   ├── Rellis-3D/                 # Required
│   ├── RUGD/                      # Optional (auto-detected)
│   └── GOOSE/                     # Optional (auto-detected)
├── checkpoints/                   # Training checkpoints
├── deploy/                        # ONNX and QNN binaries
├── configs/
│   └── rellis3d_unified.yaml
└── setup.sh                       # Environment setup
```

---

## Known Issues

### Focal Loss: Never pass alpha as `weight=` to `F.cross_entropy`

This corrupts `p_t` → `p_t^alpha`, breaking focal modulation. Alpha must be applied separately after computing unweighted CE. See `src/dataset.py` for the correct implementation.

### DDRNet output resolution may differ from input

Some DDRNet variants output at 1/8 resolution. The wrapper and inference scripts handle upsampling, but custom code must check:

```python
logits = model(x)                  # may be (1, 7, 68, 80)
logits = F.interpolate(logits, size=(544, 640), mode='bilinear')
pred = logits.argmax(dim=1)
```

### Qualcomm AI Hub device name

IQ-9075 may not appear directly in the AI Hub device catalog. Use:

```python
device = hub.Device("Qualcomm QCS9075 (Proxy)")
```

---

## License

Training code (`scripts/`, `src/`, `setup.sh`) — **MIT License**.

| Component | License |
|-----------|---------|
| DDRNet (model & weights) | MIT |
| qai_hub_models | BSD-3 |
| EfficientViT (model & weights) | Apache 2.0 |
| RELLIS-3D (dataset) | CC BY-NC-SA 3.0 |
| RUGD (dataset) | CC BY 4.0 |
| GOOSE (dataset) | CC BY-SA 4.0 |

**Note:** RELLIS-3D's CC BY-NC-SA 3.0 prohibits commercial use. Weights trained with RELLIS-3D inherit this restriction.

---

## References

```bibtex
@article{hong2021deep,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv preprint arXiv:2101.06085}, year={2021}
}
@inproceedings{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={ICCV}, year={2017}
}
@inproceedings{cai2023efficientvit,
  title={EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={ICCV}, year={2023}
}
@inproceedings{jiang2021rellis3d,
  title={RELLIS-3D Dataset: Data, Benchmarks and Analysis},
  author={Jiang, Peng and Osteen, Philip and Wigness, Maggie and Saripalli, Srikanth},
  booktitle={ICRA}, year={2021}
}
@inproceedings{wigness2019rugd,
  title={A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments},
  author={Wigness, Maggie and Eum, Sungmin and Rogers, John G and Han, David and Kwon, Heesung},
  booktitle={IROS}, year={2019}
}
@inproceedings{mortimer2024goose,
  title={The GOOSE Dataset for Perception in Unstructured Environments},
  author={Mortimer, Peter and others},
  booktitle={ICRA}, year={2024}
}
```