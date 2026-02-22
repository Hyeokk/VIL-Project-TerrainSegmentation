#!/usr/bin/env python3
"""
Environment and dataset verification for the unified 7-class off-road
segmentation project.

Primary model: DDRNet23-Slim (via qai_hub_models, INT8-safe)
Legacy models: EfficientViT-B0/B1, FFNet (optional)

Verifies:
  1. GPU & PyTorch
  2. RELLIS-3D dataset directory
  3. RUGD / GOOSE dataset directories
  4. Split files
  5. Label values (7-class unified ontology)
  6. Model loading (DDRNet23-Slim primary, legacy optional)
  7. Forward pass + speed benchmark
  8. Training configuration checklist

Usage:
    conda activate offroad
    python scripts/verify_all.py
"""

import os
import sys
import time
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Rellis-3D")
RUGD_ROOT = os.path.join(PROJECT_ROOT, "data", "RUGD")
GOOSE_ROOT = os.path.join(PROJECT_ROOT, "data", "GOOSE")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "Rellis-3D", "split")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.append(PROJECT_ROOT)

# 7-class unified ontology mapping
from src.dataset import RELLIS_TO_UNIFIED, NUM_CLASSES, CLASS_NAMES

errors = []
warnings = []


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg):
    print(f"  [ERR]  {msg}")
    errors.append(msg)


def warn(msg):
    print(f"  [WARN] {msg}")
    warnings.append(msg)


print()
print("Paths")
print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
print(f"  DATA_ROOT    : {DATA_ROOT}")
print(f"  RUGD_ROOT    : {RUGD_ROOT}")
print(f"  GOOSE_ROOT   : {GOOSE_ROOT}")
print()

# ======================================================================
# 1) GPU & PyTorch
# ======================================================================
print("=" * 60)
print("[1/8] GPU & PyTorch")
print("=" * 60)
try:
    import torch
    import torchvision

    ok(f"PyTorch {torch.__version__}")
    ok(f"Torchvision {torchvision.__version__}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"GPU: {gpu} (CC {cc[0]}.{cc[1]}, {mem:.1f} GB)")
        arch_list = torch.cuda.get_arch_list()
        if "sm_120" in arch_list:
            ok("Blackwell sm_120 support detected")
        else:
            warn(f"sm_120 not in arch list {arch_list} (may work on non-Blackwell GPU)")
        x = torch.randn(100, 100).cuda()
        _ = torch.matmul(x, x)
        ok("Basic GPU compute test passed")
    else:
        fail("CUDA is not available")
except Exception as e:
    fail(f"PyTorch error: {e}")

# ======================================================================
# 2) RELLIS-3D dataset directory
# ======================================================================
print()
print("=" * 60)
print("[2/8] RELLIS-3D directory")
print("=" * 60)

total_images = 0
total_labels = 0
if os.path.exists(DATA_ROOT):
    ok(f"Data root exists: {DATA_ROOT}")
    sequences = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("0000")
    )
    if sequences:
        ok(f"Sequence folders: {sequences}")
    else:
        fail("No sequence folders (00000~00004) found")

    for seq in sequences:
        img_dir = os.path.join(DATA_ROOT, seq, "pylon_camera_node")
        lbl_dir = os.path.join(DATA_ROOT, seq, "pylon_camera_node_label_id")
        if os.path.exists(img_dir):
            imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
            total_images += len(imgs)
            ok(f"  {seq}/images: {len(imgs)}")
        else:
            fail(f"  {seq}/pylon_camera_node not found")
        if os.path.exists(lbl_dir):
            lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".png")]
            total_labels += len(lbls)
            ok(f"  {seq}/labels: {len(lbls)}")
        else:
            fail(f"  {seq}/pylon_camera_node_label_id not found")

    ok(f"Total: {total_images} images / {total_labels} labels")
else:
    fail(f"Data root not found: {DATA_ROOT}")

# ======================================================================
# 3) RUGD / GOOSE directories
# ======================================================================
print()
print("=" * 60)
print("[3/8] RUGD & GOOSE directories (optional)")
print("=" * 60)

# RUGD
if os.path.isdir(RUGD_ROOT):
    frames_dir = os.path.join(RUGD_ROOT, "RUGD_frames-with-annotations")
    annot_dir = os.path.join(RUGD_ROOT, "RUGD_annotations")
    if os.path.isdir(frames_dir) and os.path.isdir(annot_dir):
        scene_count = len([d for d in os.listdir(frames_dir)
                          if os.path.isdir(os.path.join(frames_dir, d))])
        ok(f"RUGD found: {scene_count} scenes in {RUGD_ROOT}")
    else:
        warn(f"RUGD root exists but missing RUGD_frames-with-annotations or RUGD_annotations")
else:
    warn(f"RUGD not found at {RUGD_ROOT} (optional, training will use RELLIS-3D only)")

# GOOSE
if os.path.isdir(GOOSE_ROOT):
    # Check common layouts
    goose_ok = False
    for img_candidate in [
        os.path.join(GOOSE_ROOT, "images", "train"),
        os.path.join(GOOSE_ROOT, "train", "images", "train"),
    ]:
        if os.path.isdir(img_candidate):
            scene_count = len([d for d in os.listdir(img_candidate)
                              if os.path.isdir(os.path.join(img_candidate, d))])
            ok(f"GOOSE found: {scene_count} scenes in {GOOSE_ROOT}")
            goose_ok = True
            break
    if not goose_ok:
        warn(f"GOOSE root exists but image directory layout not recognized")

    # Check label mapping CSV
    csv_found = False
    for csv_candidate in [
        os.path.join(GOOSE_ROOT, "goose_label_mapping.csv"),
        os.path.join(GOOSE_ROOT, "train", "goose_label_mapping.csv"),
    ]:
        if os.path.isfile(csv_candidate):
            ok(f"GOOSE label mapping CSV found: {csv_candidate}")
            csv_found = True
            break
    if not csv_found:
        warn("goose_label_mapping.csv not found (GOOSE will use hardcoded mapping)")
else:
    warn(f"GOOSE not found at {GOOSE_ROOT} (optional)")

# ======================================================================
# 4) Split files
# ======================================================================
print()
print("=" * 60)
print("[4/8] Split files")
print("=" * 60)

for split_name in ["train.lst", "val.lst", "test.lst"]:
    split_path = os.path.join(SPLIT_DIR, split_name)
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        ok(f"{split_name}: {len(lines)} entries")
    else:
        warn(f"{split_name} not found: {split_path}")

custom_dir = os.path.join(DATA_ROOT, "split_custom")
for split_name in ["train_70.lst", "test_30.lst"]:
    split_path = os.path.join(custom_dir, split_name)
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        ok(f"custom/{split_name}: {len(lines)} entries")
    else:
        warn(f"custom/{split_name} not found (run: python scripts/make_split_custom.py)")

# ======================================================================
# 5) Label values (7-class unified)
# ======================================================================
print()
print("=" * 60)
print("[5/8] Label values -- 7-class unified ontology")
print("=" * 60)
try:
    train_lst = os.path.join(SPLIT_DIR, "train.lst")
    if os.path.exists(train_lst):
        with open(train_lst, "r") as f:
            first_line = f.readline().strip().split()
        lbl_path = os.path.join(DATA_ROOT, first_line[1])
        lbl = np.array(Image.open(lbl_path))
        unique_ids = sorted(np.unique(lbl).tolist())
        ok(f"Label shape: {lbl.shape}, dtype: {lbl.dtype}")
        ok(f"Unique original IDs: {unique_ids}")

        known_ids = set(RELLIS_TO_UNIFIED.keys())
        unknown = [v for v in unique_ids if v not in known_ids]
        if not unknown:
            ok("All label IDs covered by RELLIS_TO_UNIFIED mapping")
        else:
            warn(f"Unknown IDs (will be ignored=255): {unknown}")

        # Verify remapping produces valid 7-class output
        remapped = np.full_like(lbl, 255, dtype=np.int64)
        for orig_id, unified_id in RELLIS_TO_UNIFIED.items():
            remapped[lbl == orig_id] = unified_id
        unified_ids = sorted(np.unique(remapped).tolist())
        ok(f"Remapped unified IDs: {unified_ids}")
        valid_ids = set(range(NUM_CLASSES)) | {255}
        invalid = [v for v in unified_ids if v not in valid_ids]
        if not invalid:
            ok(f"All remapped IDs valid for {NUM_CLASSES}-class ontology")
        else:
            fail(f"Invalid remapped IDs: {invalid}")

        ok(f"Ontology: {NUM_CLASSES} classes = {CLASS_NAMES}")
    else:
        warn("Cannot check labels -- train.lst not found")
except Exception as e:
    fail(f"Label check failed: {e}")

# ======================================================================
# 6) Model loading -- DDRNet23-Slim (primary) + legacy (optional)
# ======================================================================
print()
print("=" * 60)
print("[6/8] Model loading -- DDRNet23-Slim (primary)")
print("=" * 60)

model_for_speed = None

try:
    from src.models import build_model, SUPPORTED_MODELS

    ok(f"models.py loaded, {len(SUPPORTED_MODELS)} models registered")

    # --- DDRNet23-Slim (PRIMARY) ---
    try:
        model = build_model("ddrnet23-slim", num_classes=NUM_CLASSES, pretrained=True)
        params = sum(p.numel() for p in model.parameters())
        ok(f"ddrnet23-slim: {params:,} params ({params/1e6:.2f}M) [PRIMARY]")
        model_for_speed = model
    except Exception as e:
        fail(f"ddrnet23-slim load failed: {e}")

    # --- EfficientViT-B1 (LEGACY, optional) ---
    try:
        model_b1 = build_model("efficientvit-b1", num_classes=NUM_CLASSES, pretrained=True)
        params = sum(p.numel() for p in model_b1.parameters())
        ok(f"efficientvit-b1: {params:,} params ({params/1e6:.2f}M) [legacy]")
        del model_b1
    except Exception as e:
        warn(f"efficientvit-b1 not available (legacy, optional): {e}")

    # --- FFNet (LEGACY, optional) ---
    try:
        model_ff = build_model("ffnet-78s", num_classes=NUM_CLASSES, pretrained=True)
        params = sum(p.numel() for p in model_ff.parameters())
        ok(f"ffnet-78s: {params:,} params ({params/1e6:.2f}M) [legacy]")
        del model_ff
    except Exception as e:
        warn(f"ffnet-78s not available (legacy, optional)")

except Exception as e:
    fail(f"Model factory import failed: {e}")

# ======================================================================
# 7) Forward pass + speed
# ======================================================================
print()
print("=" * 60)
print("[7/8] Forward pass and inference speed")
print("=" * 60)

if model_for_speed is not None:
    try:
        model_for_speed = model_for_speed.cuda().eval()
        dummy = torch.randn(1, 3, 544, 640).cuda()

        with torch.no_grad():
            output = model_for_speed(dummy)
        ok(f"Input: {list(dummy.shape)} -> Output: {list(output.shape)}")

        if output.shape[1] == NUM_CLASSES:
            ok(f"Output classes: {output.shape[1]} (expected: {NUM_CLASSES})")
        else:
            fail(f"Output classes mismatch: {output.shape[1]} != {NUM_CLASSES}")

        # Speed benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                _ = model_for_speed(dummy)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        avg_ms = np.mean(times[20:]) * 1000
        fps = 1000 / avg_ms
        ok(f"DDRNet23-Slim speed: {avg_ms:.2f} ms ({fps:.1f} FPS) @ 544x640")

        del model_for_speed, dummy
        torch.cuda.empty_cache()
    except Exception as e:
        fail(f"Forward pass failed: {e}")
else:
    warn("Skipping speed test -- no model loaded")

# ======================================================================
# 8) Training configuration checklist
# ======================================================================
print()
print("=" * 60)
print("[8/8] Training configuration checklist")
print("=" * 60)

checklist = {
    "Primary model": "DDRNet23-Slim (5.7M params, INT8-safe, Qualcomm verified)",
    "Legacy models": "EfficientViT-B0/B1, FFNet (optional backup)",
    "Ontology":      f"Unified {NUM_CLASSES}-class (caterpillar-aware)",
    "Datasets":      "RELLIS-3D (required) + RUGD + GOOSE (auto-detected)",
    "Optimizer":     "AdamW (lr=0.001, wd=0.01)",
    "Scheduler":     "LinearLR warmup 20ep + CosineAnnealing 180ep",
    "Loss":          "Focal Loss (gamma=2.0, per-class weights)",
    "EMA":           "decay=0.9999",
    "Augmentation":  "Flip, MultiScaleCrop, ColorJitter, GaussBlur, Shadow, Erasing",
    "Grad Clipping": "max_norm=5.0",
    "Deploy target": "Qualcomm IQ-9075 (100 TOPS NPU, INT8)",
    "Deploy path":   "PyTorch -> ONNX -> QNN Context Binary -> NPU",
}

print()
for k, v in checklist.items():
    print(f"  {k:16s} | {v}")
print()

# ======================================================================
# Fast mode check
# ======================================================================
print("=" * 60)
print("Fast mode data")
print("=" * 60)
for name, path in [
    ("Rellis-3D_fast", os.path.join(PROJECT_ROOT, "data", "Rellis-3D_fast")),
    ("RUGD_fast", os.path.join(PROJECT_ROOT, "data", "RUGD_fast")),
    ("GOOSE_fast", os.path.join(PROJECT_ROOT, "data", "GOOSE_fast")),
]:
    if os.path.isdir(path):
        ok(f"{name} exists (--fast mode ready)")
    else:
        warn(f"{name} not found (run: python scripts/preprocess_datasets.py)")

# ======================================================================
# Summary
# ======================================================================
print()
print("=" * 60)
print(f"Result: {len(errors)} errors, {len(warnings)} warnings")
print("=" * 60)

if errors:
    print("\n[Errors -- must fix]")
    for e in errors:
        print(f"  * {e}")

if warnings:
    print("\n[Warnings -- optional]")
    for w in warnings:
        print(f"  - {w}")

if not errors:
    print("\nEnvironment ready. Start training with:")
    print("  python scripts/train.py --model ddrnet23-slim --fast --num_workers 8")
    print()
    print("Legacy models (optional):")
    print("  python scripts/train.py --model efficientvit-b1 --fast --num_workers 8")
else:
    print("\nPlease fix errors above and re-run verification.")