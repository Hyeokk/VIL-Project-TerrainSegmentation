# Off-Road Semantic Segmentation for Tracked Robot

Semantic segmentation model trained on three open-source off-road datasets (RELLIS-3D, RUGD, GOOSE) under a unified 7-class ontology for real-time traversability estimation on a caterpillar-track autonomous robot.

For deployment (ONNX export, INT8 quantization, ROS2 inference), see [VIL-Project-AMR (amr-segmentation branch)](https://github.com/Hyeokk/VIL-Project-AMR/tree/amr-segmentation).

<table>
<tr>
<td align="center"><img src="docs/road_demo.gif" width="400"><br>Road</td>
<td align="center"><img src="docs/sidewalk_demo.gif" width="400"><br>Sidewalk</td>
</tr>
</table>

> Smooth Ground occasionally flickers to Obstacle. This is caused by the domain gap between training datasets (RELLIS-3D, RUGD, GOOSE) and the target camera (S10 Ultra), as no target-domain data was used during training.

---

## Model Selection

| | EfficientViT-B1 | **DDRNet23-Slim** |
|---|---|---|
| Architecture | Vision Transformer | CNN (Dual-Resolution) |
| Parameters | 4.8M | 5.7M |
| Cityscapes mIoU (FP32) | 80.5% | 77.8% |
| Core operations | Softmax, LayerNorm, GELU | Conv + BatchNorm + ReLU |
| Pretrained source | MIT official weights | Qualcomm AI Hub (qai_hub_models) |
| NPU compatibility | Partial (unsupported ops) | Full (131/131 ops on NPU) |

DDRNet23-Slim uses only Conv + BN + ReLU, making it inherently INT8-safe and fully compatible with the Qualcomm Hexagon NPU. EfficientViT-B1 achieves higher FP32 accuracy but fails under INT8 quantization. See [docs/model_selection.md](docs/model_selection.md) for detailed analysis.

---

## 7-Class Ontology

| ID | Class | Examples | Traversability |
|----|-------|----------|----------------|
| 0 | Smooth Ground | Asphalt, concrete, packed dirt | Optimal path |
| 1 | Rough Ground | Sand, gravel, mud, snow | Passable (reduce speed) |
| 2 | Vegetation | Low grass, moss, leaves | Passable (tracks) |
| 3 | Obstacle | Trees, rocks, buildings, fences, bushes | Avoid |
| 4 | Water | Puddles, streams, lakes | Avoid (flood risk) |
| 5 | Sky | Sky, clouds | Ignore |
| 6 | Dynamic | People, vehicles, animals | Avoid (safety) |

Mappings specific to caterpillar-track robots: bush is classified as Obstacle due to track entanglement risk, puddle as Water due to unknown depth, dirt as Smooth Ground as it is an optimal surface for tracks, and grass as Vegetation since low grass is traversable. Full per-dataset mappings in `src/dataset.py`.

---

## Datasets

| Dataset | Images | Environment | License |
|---------|--------|-------------|---------|
| RELLIS-3D | 4,169 | Military trails (US) | CC BY-NC-SA 3.0 |
| RUGD | 7,436 | Parks, forests | CC BY 4.0 |
| GOOSE | 7,845 | European outdoor | CC BY-SA 4.0 |
| **Total** | **19,450** | | |

Validation: RELLIS-3D 30% test split (1,788 images). RELLIS-3D is required; RUGD and GOOSE are optional and auto-detected at runtime.

---

## Training

### Quick Start

```bash
bash setup.sh && conda activate offroad
python scripts/make_split_custom.py
python scripts/preprocess_datasets.py        # fast mode, one-time
python scripts/verify_all.py
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | DDRNet23-Slim (5.7M), Cityscapes pretrained |
| Input | 544 x 640 (H x W), 0.5x of S10 Ultra camera |
| Batch / Epochs | 8 / 200 |
| Optimizer | AdamW (weight_decay=0.01) |
| Backbone LR | 1e-4 (10x lower, preserves pretrained features) |
| Head LR | 1e-3 (fast convergence for 7-class head) |
| LR Schedule | 5-ep warmup, 195-ep cosine annealing |
| Loss | Focal Loss (gamma=2.0) |
| Loss Alpha | Water=5.0, Dynamic=5.0, Rough=3.0, Smooth=1.5, others=1.0 |
| EMA | decay=0.9999 |
| AMP | FP16 mixed precision |
| Grad Clip | max_norm=5.0 |

The differential learning rate between backbone and head is critical. A uniform lr=1e-3 destroys backbone features within 2 epochs, causing mode collapse. Adjustable via `--bb_lr_factor`.

### Augmentation

Training applies spatial and photometric augmentations to address scale variation and camera domain gap. Validation uses only resize and normalize. Geometric distortion is not applied (handled by camera calibration at deployment).

| Augmentation | Parameters | Stage |
|-------------|-----------|-------|
| RandomScale | 0.5x - 2.0x | Train |
| Pad + RandomCrop | 544 x 640 | Train |
| Random Horizontal Flip | p=0.5 | Train |
| ColorJitter | brightness/contrast/saturation=0.4, hue=0.15 | Train |
| Gaussian Blur | p=0.3, radius=0.5-2.0 | Train |
| Random Shadow | p=0.2, darkening=0.3-0.7 | Train |
| Random Erasing | p=0.3, scale=0.02-0.2 | Train |
| Random Grayscale | p=0.05 | Train |
| Resize | 544 x 640 | Val |
| ImageNet Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Both |

---

## Inference (Host PC)

```bash
python scripts/infer_cam.py --checkpoint best_model.pth --input video.mp4 --output result.mp4 --overlay
```

For IQ-9075 deployment, see [VIL-Project-AMR (amr-segmentation branch)](https://github.com/Hyeokk/VIL-Project-AMR/tree/amr-segmentation).

---

## Project Structure

```
VIL-Project-TerrainSegmentation/
├── scripts/
│   ├── train.py                # Training
│   ├── evaluate.py             # mIoU evaluation
│   ├── infer_cam.py            # Host PC inference
│   ├── export_onnx.py          # ONNX export
│   ├── export_qnn.py           # QNN export
│   ├── preprocess_datasets.py  # Fast mode
│   ├── make_split_custom.py    # Train/test split
│   └── verify_all.py           # Environment check
├── src/
│   ├── dataset.py              # Datasets, mappings, Focal Loss, EMA
│   ├── models.py               # Model factory
│   └── models_ddrnet.py        # DDRNet builder (qai_hub_models)
├── data/                       # Datasets (download separately)
├── checkpoints/
├── deploy/
├── docs/
│   └── model_selection.md      # EfficientViT failure + DDRNet analysis
└── setup.sh
```

---

## Training Environment

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX PRO 4000 Blackwell (24GB) |
| CPU / RAM | AMD Ryzen 9700X / 64GB DDR5 |
| Framework | PyTorch 2.10.0, CUDA 12.8 |
| Camera | MRDVS S10 Ultra (1280 x 1080) |
| Deploy target | Qualcomm IQ-9075 (100 TOPS NPU) |

---

## License

Training code: **MIT License**.

| Component | License |
|-----------|---------|
| DDRNet | MIT |
| qai_hub_models | BSD-3 |
| EfficientViT | Apache 2.0 |
| RELLIS-3D | CC BY-NC-SA 3.0 |
| RUGD | CC BY 4.0 |
| GOOSE | CC BY-SA 4.0 |

Weights trained with RELLIS-3D inherit the CC BY-NC-SA 3.0 non-commercial restriction.

---

## References

```bibtex
@article{hong2021ddrnet,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv:2101.06085}, year={2021}
}
@inproceedings{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and others}, booktitle={ICCV}, year={2017}
}
@inproceedings{cai2023efficientvit,
  title={EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction},
  author={Cai, Han and others}, booktitle={ICCV}, year={2023}
}
@inproceedings{jiang2021rellis3d,
  title={RELLIS-3D Dataset}, author={Jiang, Peng and others}, booktitle={ICRA}, year={2021}
}
@inproceedings{wigness2019rugd,
  title={RUGD Dataset}, author={Wigness, Maggie and others}, booktitle={IROS}, year={2019}
}
@inproceedings{mortimer2024goose,
  title={The GOOSE Dataset}, author={Mortimer, Peter and others}, booktitle={ICRA}, year={2024}
}
```