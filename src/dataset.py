import os
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF


# ===================================================================
# Unified 7-class Traversability Ontology (Caterpillar-track robot)
# ===================================================================
# Designed for a caterpillar-type wheeled robot. Key design decisions:
#   - bush is classified as Obstacle (class 3), NOT Vegetation,
#     because tall/dense bushes impede caterpillar track mobility
#   - low grass is Vegetation (traversable), but high vegetation
#     and bush are considered obstacles
#   - puddle is Water/Hazard (class 4) since caterpillar tracks
#     can get stuck in wet muddy terrain
#
#   0: Smooth Ground  -- free to traverse (dirt, asphalt, concrete)
#   1: Rough Ground   -- traversable with caution (mud, rubble, gravel, sand)
#   2: Vegetation     -- low vegetation, generally traversable (short grass)
#   3: Obstacle       -- non-traversable (tree, bush, pole, fence, building, log, rock)
#   4: Water          -- hazardous for caterpillar tracks
#   5: Sky            -- not applicable for navigation costmap
#   6: Dynamic        -- moving objects (person, vehicle, animal)
# 255: Ignore         -- void / unlabeled
# ===================================================================
NUM_CLASSES = 7

CLASS_NAMES = [
    "Smooth Ground",   # 0
    "Rough Ground",    # 1
    "Vegetation",      # 2
    "Obstacle",        # 3
    "Water",           # 4
    "Sky",             # 5
    "Dynamic",         # 6
]

# Color palette for visualization (RGB)
# Each class has a distinct, easily distinguishable color
CLASS_COLORS = [
    (128, 64, 128),    # 0: Smooth Ground -- purple-gray (road-like)
    (140, 100, 40),    # 1: Rough Ground  -- brown
    (0, 180, 0),       # 2: Vegetation    -- green
    (220, 20, 60),     # 3: Obstacle      -- crimson red
    (0, 100, 255),     # 4: Water         -- blue
    (70, 130, 180),    # 5: Sky           -- steel blue
    (255, 255, 0),     # 6: Dynamic       -- yellow
]


# -------------------------------------------------------------------
# RELLIS-3D ID -> Unified 7-class
# -------------------------------------------------------------------
# RELLIS-3D ontology (20 classes, ID mapping from their GitHub):
#   0: void, 1: dirt, 3: grass, 4: tree, 5: pole, 6: water,
#   7: sky, 8: vehicle, 9: object/container, 10: asphalt,
#   12: building, 15: log, 17: person, 18: fence, 19: bush,
#   23: concrete, 27: barrier, 29: puddle, 30: mud, 31: rubble, 33: sky, 34: generic-object
# Note: RELLIS-3D label IDs are non-contiguous.
# -------------------------------------------------------------------
RELLIS_TO_UNIFIED = {
    0:  255,   # void -> ignore
    1:  0,     # dirt -> Smooth Ground  (RESTORED from ignore)
    3:  2,     # grass -> Vegetation (low grass, traversable for caterpillar)
    4:  3,     # tree -> Obstacle
    5:  3,     # pole -> Obstacle
    6:  4,     # water -> Water
    7:  5,     # sky -> Sky
    8:  6,     # vehicle -> Dynamic
    9:  3,     # object/container -> Obstacle
    10: 0,     # asphalt -> Smooth Ground
    12: 3,     # building -> Obstacle
    15: 3,     # log -> Obstacle
    17: 6,     # person -> Dynamic
    18: 3,     # fence -> Obstacle
    19: 3,     # bush -> Obstacle (caterpillar cannot push through dense bush)
    23: 0,     # concrete -> Smooth Ground
    27: 3,     # barrier -> Obstacle
    29: 4,     # puddle -> Water (caterpillar tracks risk getting stuck)
    30: 1,     # mud -> Rough Ground
    31: 1,     # rubble -> Rough Ground
    33: 5,     # sky (alternate ID) -> Sky
    34: 3,     # generic-object -> Obstacle
}


# -------------------------------------------------------------------
# RUGD color -> class name -> Unified 7-class
# -------------------------------------------------------------------
# RUGD uses color-encoded annotation PNGs (24 classes).
# Each class has a unique RGB color in the annotation image.
# We map color -> RUGD class name -> unified class ID.
# -------------------------------------------------------------------
RUGD_COLOR_MAP = {
    (0, 0, 0):       "void",
    (108, 64, 20):   "dirt",
    (255, 229, 204): "sand",
    (0, 102, 0):     "grass",
    (0, 255, 0):     "tree",
    (0, 153, 153):   "pole",
    (0, 128, 255):   "water",
    (0, 0, 255):     "sky",
    (255, 255, 0):   "vehicle",
    (255, 0, 127):   "container",
    (64, 64, 64):    "asphalt",
    (255, 128, 0):   "gravel",
    (255, 0, 0):     "building",
    (153, 76, 0):    "mulch",
    (102, 102, 0):   "rock-bed",
    (102, 0, 0):     "log",
    (0, 255, 128):   "bicycle",
    (204, 153, 255): "person",
    (102, 0, 204):   "fence",
    (255, 153, 204): "bush",
    (0, 102, 102):   "sign",
    (153, 204, 255): "rock",
    (102, 255, 255): "bridge",
    (101, 101, 11):  "concrete",
    (204, 204, 0):   "picnic-table",
}

RUGD_NAME_TO_UNIFIED = {
    "void":         255,
    "dirt":         0,     # Smooth Ground
    "sand":         1,     # Rough Ground
    "grass":        2,     # Vegetation
    "tree":         3,     # Obstacle
    "pole":         3,     # Obstacle
    "water":        4,     # Water
    "sky":          5,     # Sky
    "vehicle":      6,     # Dynamic
    "container":    3,     # Obstacle
    "asphalt":      0,     # Smooth Ground
    "gravel":       1,     # Rough Ground
    "building":     3,     # Obstacle
    "mulch":        1,     # Rough Ground
    "rock-bed":     1,     # Rough Ground
    "log":          3,     # Obstacle
    "bicycle":      6,     # Dynamic
    "person":       6,     # Dynamic
    "fence":        3,     # Obstacle
    "bush":         3,     # Obstacle (caterpillar: dense bush blocks movement)
    "sign":         3,     # Obstacle
    "rock":         3,     # Obstacle
    "bridge":       0,     # Smooth Ground
    "concrete":     0,     # Smooth Ground
    "picnic-table": 3,     # Obstacle
}


# -------------------------------------------------------------------
# GOOSE class ID -> Unified 7-class
# -------------------------------------------------------------------
# GOOSE uses 64 fine-grained classes grouped into categories.
# Labels are stored as single-channel PNG with pixel values = class ID.
# GOOSE ontology reference: https://goose-dataset.de/docs/
# We map the 64 classes into our 7-class traversability ontology.
#
# For the ICRA 2025 GOOSE Challenge, the 64 classes are often
# consolidated into 9 operational categories; we further merge
# those into our 7-class scheme.
# -------------------------------------------------------------------
GOOSE_TO_UNIFIED = {}

# Terrain / Ground surfaces -> Smooth Ground (0) or Rough Ground (1)
_goose_smooth = [
    "asphalt", "concrete", "dirt_road", "cobblestone", "cobble",
    "gravel_road", "paved_road", "sidewalk", "road",
    "bikeway", "pedestrian_crossing", "road_marking",
]
_goose_rough = [
    "sand", "gravel", "mud", "soil", "rubble", "snow", "ice",
    "dirt", "forest_floor", "leaflitter", "woodchips", "crops",
]
# Vegetation -> Vegetation (2) for low, Obstacle (3) for tall/dense
_goose_low_veg = [
    "low_grass", "leaves", "moss", "clover", "grass",
]
_goose_high_veg = [
    "high_grass", "bush", "hedge", "tree_crown", "tree_trunk",
    "shrub", "reed", "fern",
    "forest", "scenery_vegetation", "tree_root",
]
# Structures -> Obstacle (3)
_goose_obstacle = [
    "pole", "fence", "wall", "building", "rock", "boulder",
    "sign", "traffic_sign", "barrier", "guardrail", "guard_rail",
    "bollard", "curb", "stairs", "log", "stump", "root", "debris",
    "container", "barrel", "road_block", "heavy_machinery",
    "construction_sign", "trash", "tire",
    "traffic_cone", "fire_hydrant", "bench", "table", "chair",
    "bridge", "tunnel", "pipeline", "pillar", "post",
    "obstacle",  # GOOSE generic obstacle class
    "street_light", "traffic_light", "misc_sign",
    "boom_barrier", "barrier_tape", "rail_track",
    "wire", "pipe",
]
# Water -> Water (4)
_goose_water = ["puddle", "water", "stream", "river", "lake"]
# Sky -> Sky (5)
_goose_sky = ["sky", "cloud"]
# Dynamic -> Dynamic (6)
_goose_dynamic = [
    "person", "car", "truck", "bicycle", "motorcycle",
    "animal", "dog", "horse", "rider",
    "bus", "on_rails", "caravan", "trailer",
    "kick_scooter", "military_vehicle",
]
# Ignore
_goose_ignore = [
    "void", "ego_vehicle", "unknown", "other",
    "undefined", "unlabeled", "static", "dynamic",
    "outlier",
]

# Build the mapping (name-based, case-insensitive)
for _name in _goose_smooth:
    GOOSE_TO_UNIFIED[_name] = 0
for _name in _goose_rough:
    GOOSE_TO_UNIFIED[_name] = 1
for _name in _goose_low_veg:
    GOOSE_TO_UNIFIED[_name] = 2
for _name in _goose_high_veg:
    GOOSE_TO_UNIFIED[_name] = 3
for _name in _goose_obstacle:
    GOOSE_TO_UNIFIED[_name] = 3
for _name in _goose_water:
    GOOSE_TO_UNIFIED[_name] = 4
for _name in _goose_sky:
    GOOSE_TO_UNIFIED[_name] = 5
for _name in _goose_dynamic:
    GOOSE_TO_UNIFIED[_name] = 6
for _name in _goose_ignore:
    GOOSE_TO_UNIFIED[_name] = 255


# ===================================================================
# Photometric Augmentation
# ===================================================================
# Since inference will use calibrated + undistorted S10 Ultra images,
# NO geometric distortion simulation is applied.
# Focus is on color/tone domain gap and environmental variation.
# ===================================================================

class PhotometricAugmentation:
    """Strong photometric augmentation for off-road environments.

    Bridges the color/tone domain gap between open-source training
    datasets (Basler, etc.) and the MRDVS S10 Ultra camera, WITHOUT
    any geometric distortion (handled by calibration at inference).
    """

    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.15,
        )

    def __call__(self, image):
        # ColorJitter: brightness, contrast, saturation, hue
        image = self.color_jitter(image)

        # Gaussian blur for motion blur / defocus simulation (p=0.3)
        if random.random() < 0.3:
            radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # Random grayscale for washed-out lighting conditions (p=0.05)
        if random.random() < 0.05:
            image = TF.to_grayscale(image, num_output_channels=3)

        # Simulated shadow patches from tree canopy (p=0.2)
        if random.random() < 0.2:
            image = self._apply_random_shadow(image)

        return image

    @staticmethod
    def _apply_random_shadow(image):
        """Darken a random rectangular region to simulate tree shadows."""
        w, h = image.size
        img_array = np.array(image, dtype=np.float32)

        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(x1 + w // 4, w)
        y2 = random.randint(y1 + h // 4, h)

        shadow_factor = random.uniform(0.4, 0.7)
        img_array[y1:y2, x1:x2] *= shadow_factor
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)


# ===================================================================
# Base Dataset Class with Shared Augmentation Logic
# ===================================================================

class BaseSegDataset(Dataset):
    """Base class with shared augmentation and preprocessing logic.

    All dataset-specific subclasses inherit from this to ensure
    consistent augmentation pipeline across RELLIS-3D, RUGD, and GOOSE.

    Supports rectangular crop sizes to match deployment aspect ratio.
    Following MMSegmentation's Cityscapes pipeline pattern:
        Resize(img_scale, ratio_range=(0.5, 2.0)) → RandomCrop(crop_h, crop_w)

    For S10 Ultra (1280×1080) deployment:
        - Deploy resolution: 640×544 (half-res, 32-pixel aligned)
        - Training crop: (544, 640) matching deploy aspect ratio
        - This ensures train/inference distribution match
    """

    # Random scale range for resize-before-crop (MMSeg standard)
    RANDOM_SCALE_RANGE = (0.5, 2.0)

    def __init__(self, is_train=True, crop_size=512):
        self.is_train = is_train

        # Support both int (square) and tuple (h, w) crop sizes
        if isinstance(crop_size, (list, tuple)):
            self.crop_h, self.crop_w = crop_size[0], crop_size[1]
        else:
            self.crop_h = self.crop_w = crop_size

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.photometric = PhotometricAugmentation()
        self.random_erasing = transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))

    def _apply_transforms(self, image, label):
        """Apply shared augmentation pipeline to image-label pair.

        Training pipeline (follows MMSeg Cityscapes standard):
            1. Random scale resize (0.5x ~ 2.0x, aspect-ratio preserving)
               → At small scale, crop covers most of image (global context)
               → At large scale, crop covers fine detail (local context)
            2. Pad if image is smaller than crop size
            3. Random horizontal flip
            4. Random crop → (crop_h × crop_w)  e.g. 544×640
            5. Photometric augmentation
            6. Random erasing

        Validation pipeline:
            1. Resize to exactly (crop_h × crop_w)  e.g. 544×640
               → Matches inference preprocessing exactly

        Args:
            image: PIL.Image (RGB)
            label: PIL.Image (uint8, pixel values = unified class IDs)

        Returns:
            image: torch.Tensor (3, crop_h, crop_w), normalized
            label: torch.Tensor (crop_h, crop_w), long
        """
        if self.is_train:
            # ----- Step 1: Random scale resize (aspect-ratio preserving) -----
            w, h = image.size
            scale = random.uniform(*self.RANDOM_SCALE_RANGE)
            new_h = int(h * scale)
            new_w = int(w * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            label = label.resize((new_w, new_h), Image.NEAREST)

            # ----- Step 2: Pad if smaller than crop size -----
            w, h = image.size
            pad_h = max(self.crop_h - h, 0)
            pad_w = max(self.crop_w - w, 0)
            if pad_h > 0 or pad_w > 0:
                # Pad image with mean color, label with ignore (255)
                image = TF.pad(image, [0, 0, pad_w, pad_h], fill=128)
                label = TF.pad(label, [0, 0, pad_w, pad_h], fill=255)

            # ----- Step 3: Random horizontal flip -----
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            # ----- Step 4: Random crop → (crop_h, crop_w) -----
            i, j, th, tw = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_h, self.crop_w)
            )
            image = TF.crop(image, i, j, th, tw)
            label = TF.crop(label, i, j, th, tw)

            # ----- Step 5: Photometric augmentation -----
            image = self.photometric(image)

        else:
            # Validation: resize to exact deploy resolution
            # This matches the inference preprocessing exactly,
            # so validation mIoU reflects real-world performance.
            image = image.resize((self.crop_w, self.crop_h), Image.BILINEAR)
            label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

        # To tensor + normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)
        label = torch.tensor(np.array(label), dtype=torch.long)

        # Random erasing (train only)
        if self.is_train:
            image = self.random_erasing(image)

        return image, label


# ===================================================================
# RELLIS-3D Dataset
# ===================================================================

class Rellis3DUnified(BaseSegDataset):
    """RELLIS-3D dataset remapped to unified 7-class ontology.

    Key changes from original repository:
      - dirt (ID 1) restored as Smooth Ground (was ignore 255)
      - bush (ID 19) mapped to Obstacle for caterpillar robot
      - All 20 classes remapped to 7-class traversability ontology
    """

    def __init__(self, data_root, split_file, is_train=True, crop_size=512):
        super().__init__(is_train=is_train, crop_size=crop_size)
        self.data_root = data_root
        self.pairs = []

        with open(split_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path = os.path.join(data_root, parts[0])
                    lbl_path = os.path.join(data_root, parts[1])
                    if os.path.exists(img_path) and os.path.exists(lbl_path):
                        self.pairs.append((img_path, lbl_path))

        print(f"[RELLIS-3D] Loaded {len(self.pairs)} pairs from {split_file}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(lbl_path))

        # Remap original IDs to unified 7-class
        label = np.full_like(mask, 255, dtype=np.int64)
        for orig_id, unified_id in RELLIS_TO_UNIFIED.items():
            label[mask == orig_id] = unified_id
        label = Image.fromarray(label.astype(np.uint8))

        return self._apply_transforms(image, label)


# ===================================================================
# RUGD Dataset
# ===================================================================

class RUGDUnified(BaseSegDataset):
    """RUGD dataset remapped to unified 7-class ontology.

    RUGD annotations are color-encoded PNGs where each pixel's RGB
    value represents a class. We convert colors -> class names ->
    unified IDs.

    Expected directory structure:
        data/RUGD/
            RUGD_frames-with-annotations/
                creek/
                    *.png (images)
                park-1/
                    *.png
                ...
            RUGD_annotations/
                creek/
                    *.png (color-coded labels)
                park-1/
                    *.png
                ...
    """

    def __init__(self, data_root, split_file=None, is_train=True, crop_size=512):
        super().__init__(is_train=is_train, crop_size=crop_size)
        self.data_root = data_root
        self.pairs = []

        # Build color -> unified ID lookup table (256x256x256 would be too large;
        # use a dict keyed by (R,G,B) tuples)
        self._color_to_id = {}
        for color, name in RUGD_COLOR_MAP.items():
            self._color_to_id[color] = RUGD_NAME_TO_UNIFIED.get(name, 255)

        if split_file and os.path.exists(split_file):
            # If a split file listing image paths is provided
            with open(split_file, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_path = os.path.join(data_root, parts[0])
                        lbl_path = os.path.join(data_root, parts[1])
                        if os.path.exists(img_path) and os.path.exists(lbl_path):
                            self.pairs.append((img_path, lbl_path))
        else:
            # Auto-discover from directory structure
            frames_dir = os.path.join(data_root, "RUGD_frames-with-annotations")
            annot_dir = os.path.join(data_root, "RUGD_annotations")

            if not os.path.isdir(frames_dir) or not os.path.isdir(annot_dir):
                print(f"[RUGD] Warning: directories not found at {data_root}")
                return

            for seq_name in sorted(os.listdir(frames_dir)):
                seq_frames = os.path.join(frames_dir, seq_name)
                seq_annots = os.path.join(annot_dir, seq_name)
                if not os.path.isdir(seq_frames) or not os.path.isdir(seq_annots):
                    continue

                for img_name in sorted(os.listdir(seq_frames)):
                    ext = os.path.splitext(img_name)[1].lower()
                    if ext not in (".png", ".jpg", ".jpeg"):
                        continue
                    img_path = os.path.join(seq_frames, img_name)
                    # Label is always .png, image may be .jpg (fast mode)
                    lbl_name = os.path.splitext(img_name)[0] + ".png"
                    lbl_path = os.path.join(seq_annots, lbl_name)
                    if os.path.exists(lbl_path):
                        self.pairs.append((img_path, lbl_path))

        print(f"[RUGD] Loaded {len(self.pairs)} pairs from {data_root}")

    def __len__(self):
        return len(self.pairs)

    def _convert_color_label(self, color_label_np):
        """Convert color-encoded RUGD annotation to unified class IDs.

        Args:
            color_label_np: numpy array (H, W, 3) RGB color annotation

        Returns:
            numpy array (H, W) with unified class IDs
        """
        h, w = color_label_np.shape[:2]
        label = np.full((h, w), 255, dtype=np.int64)

        for color, class_id in self._color_to_id.items():
            mask = np.all(color_label_np == np.array(color, dtype=np.uint8), axis=-1)
            label[mask] = class_id

        return label

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        color_label = np.array(Image.open(lbl_path).convert("RGB"))

        # Convert color annotation to unified class IDs
        label_np = self._convert_color_label(color_label)
        label = Image.fromarray(label_np.astype(np.uint8))

        return self._apply_transforms(image, label)


# ===================================================================
# GOOSE Dataset
# ===================================================================

class GOOSEUnified(BaseSegDataset):
    """GOOSE dataset remapped to unified 7-class ontology.

    GOOSE labels are single-channel PNG files where pixel values
    represent class IDs from the GOOSE 64-class ontology.

    Supports multiple directory layouts automatically:

    Layout A (official GOOSE):
        data/GOOSE/
            goose_label_mapping.csv
            images/train/<scene>/*_color.png
            labels/train/<scene>/*_labelids.png

    Layout B (user downloaded as single split archive):
        data/GOOSE/train/
            goose_label_mapping.csv
            images/train/<scene>/*_color.png     ← nested train/train
            labels/train/<scene>/*_labelids.png

    Layout C (flat):
        data/GOOSE/train/
            images/<scene>/*_color.png
            labels/<scene>/*_labelids.png

    Image files:   *_color.png
    Label files:   *_labelids.png (same prefix)
    """

    def __init__(self, data_root, split="train", is_train=True, crop_size=512,
                 goose_id_to_name=None):
        super().__init__(is_train=is_train, crop_size=crop_size)
        self.data_root = data_root
        self.pairs = []

        # ---------------------------------------------------------------
        # Build GOOSE ID → unified 7-class mapping
        # ---------------------------------------------------------------
        self._goose_id_to_unified = {}

        if goose_id_to_name is not None:
            # Explicitly provided mapping
            for gid, gname in goose_id_to_name.items():
                gname_lower = gname.lower().replace(" ", "_")
                self._goose_id_to_unified[gid] = GOOSE_TO_UNIFIED.get(gname_lower, 255)
        else:
            # Auto-detect goose_label_mapping.csv
            csv_path = self._find_label_mapping_csv(data_root)
            if csv_path:
                self._goose_id_to_unified = self._parse_label_mapping_csv(csv_path)
                print(f"[GOOSE] Loaded label mapping from {csv_path} "
                      f"({len(self._goose_id_to_unified)} classes)")

        # ---------------------------------------------------------------
        # Discover image-label pairs
        # ---------------------------------------------------------------
        img_dir, lbl_dir = self._find_img_lbl_dirs(data_root, split)

        if img_dir and lbl_dir:
            self._collect_pairs_recursive(img_dir, lbl_dir)

        print(f"[GOOSE] Loaded {len(self.pairs)} pairs from {data_root} (split={split})")

    @staticmethod
    def _find_label_mapping_csv(data_root):
        """Search for goose_label_mapping.csv in data_root and parent dirs."""
        candidates = [
            os.path.join(data_root, "goose_label_mapping.csv"),
            os.path.join(data_root, "train", "goose_label_mapping.csv"),
            os.path.join(data_root, "..", "goose_label_mapping.csv"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def _parse_label_mapping_csv(self, csv_path):
        """Parse goose_label_mapping.csv → {int_id: unified_class_id}.

        Expected CSV format (GOOSE official):
            id,name,category,...
            0,void,void,...
            1,asphalt,ground,...
            ...
        """
        import csv
        id_to_unified = {}
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                # Find the id and name columns (flexible header matching)
                fieldnames_lower = {fn.lower().strip(): fn for fn in reader.fieldnames}
                id_col = (fieldnames_lower.get("id") or fieldnames_lower.get("class_id")
                          or fieldnames_lower.get("labelid") or fieldnames_lower.get("label_key"))
                name_col = (fieldnames_lower.get("name") or fieldnames_lower.get("class_name")
                            or fieldnames_lower.get("label"))

                if not id_col or not name_col:
                    print(f"[GOOSE] Warning: Could not find id/name columns in {csv_path}")
                    print(f"[GOOSE]   Available columns: {reader.fieldnames}")
                    return {}

                for row in reader:
                    try:
                        gid = int(row[id_col].strip())
                        gname = row[name_col].strip().lower().replace(" ", "_").replace("-", "_")
                        unified = GOOSE_TO_UNIFIED.get(gname, 255)
                        id_to_unified[gid] = unified
                    except (ValueError, KeyError):
                        continue

        except Exception as e:
            print(f"[GOOSE] Warning: Failed to parse {csv_path}: {e}")
            return {}

        return id_to_unified

    @staticmethod
    def _find_img_lbl_dirs(data_root, split):
        """Auto-detect image and label directories for various layouts.

        Returns (img_dir, lbl_dir) or (None, None) if not found.
        """
        # Try multiple possible layouts in priority order
        candidates = [
            # Layout B: data/GOOSE/train/images/train/<scenes>/
            #   (downloaded archive extracts with nested split)
            (os.path.join(data_root, split, "images", split),
             os.path.join(data_root, split, "labels", split)),

            # Layout A: data/GOOSE/images/train/<scenes>/
            #   (official GOOSE structure)
            (os.path.join(data_root, "images", split),
             os.path.join(data_root, "labels", split)),

            # Layout C: data/GOOSE/train/images/<scenes>/
            (os.path.join(data_root, split, "images"),
             os.path.join(data_root, split, "labels")),

            # Layout D: data/GOOSE/train/images/rgb/<scenes>/
            (os.path.join(data_root, split, "images", "rgb"),
             os.path.join(data_root, split, "labels", "semantic")),
        ]

        for img_dir, lbl_dir in candidates:
            if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
                print(f"[GOOSE] Found images: {img_dir}")
                print(f"[GOOSE] Found labels: {lbl_dir}")
                return img_dir, lbl_dir

        print(f"[GOOSE] Warning: Could not find image/label dirs in {data_root}")
        print(f"[GOOSE]   Tried: {[c[0] for c in candidates]}")
        return None, None

    def _collect_pairs_recursive(self, img_dir, lbl_dir):
        """Recursively collect image-label pairs from scene subfolders.

        GOOSE naming convention:
            Image: <prefix>_color.png
            Label: <prefix>_labelids.png
        """
        # Check if img_dir contains scene subfolders or direct images
        entries = sorted(os.listdir(img_dir))
        has_subdirs = any(
            os.path.isdir(os.path.join(img_dir, e)) for e in entries
        )

        if has_subdirs:
            # Recurse into scene subfolders
            for scene in sorted(entries):
                scene_img_dir = os.path.join(img_dir, scene)
                scene_lbl_dir = os.path.join(lbl_dir, scene)
                if os.path.isdir(scene_img_dir) and os.path.isdir(scene_lbl_dir):
                    self._match_pairs_in_folder(scene_img_dir, scene_lbl_dir)
                elif os.path.isdir(scene_img_dir):
                    # Label folder might not exist for some scenes
                    pass
        else:
            # Flat directory (images directly in img_dir)
            self._match_pairs_in_folder(img_dir, lbl_dir)

    def _match_pairs_in_folder(self, img_folder, lbl_folder):
        """Match GOOSE images with *_labelids.png labels in one folder.

        GOOSE actual naming patterns:
            Images: <prefix>_windshield_vis.png  (visible RGB — original)
                    <prefix>_windshield_vis.jpg  (visible RGB — fast mode)
                    <prefix>_windshield_nir.*     (near-infrared — skip)
            Labels: <prefix>_labelids.png
                    <prefix>_color.png           (color visualization — skip)
                    <prefix>_instanceids.png     (instance IDs — skip)

        Common prefix: e.g. "2022-07-22_flight__0000_1658492967230070008"
        """
        for fname in sorted(os.listdir(img_folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".png", ".jpg", ".jpeg"):
                continue

            img_path = os.path.join(img_folder, fname)
            if not os.path.isfile(img_path):
                continue

            # Skip NIR and non-image label files
            if "_windshield_nir" in fname or "_instanceids" in fname:
                continue

            # Extract the common prefix to find matching label
            # Image: <prefix>_windshield_vis.{png,jpg} → Label: <prefix>_labelids.png
            # Image: <prefix>_color.{png,jpg}           → Label: <prefix>_labelids.png
            lbl_name = None
            if "_windshield_vis" in fname:
                prefix = fname.split("_windshield_vis")[0]
                lbl_name = prefix + "_labelids.png"
            elif "_color" in fname:
                prefix = fname.split("_color")[0]
                lbl_name = prefix + "_labelids.png"
            else:
                # Fallback: try common suffixes
                stem = os.path.splitext(fname)[0]
                for suffix in ["_labelids.png", "_label.png", "_semantic.png"]:
                    candidate = stem + suffix
                    if os.path.exists(os.path.join(lbl_folder, candidate)):
                        lbl_name = candidate
                        break
                if lbl_name is None:
                    continue

            lbl_path = os.path.join(lbl_folder, lbl_name)
            if os.path.isfile(lbl_path):
                self.pairs.append((img_path, lbl_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(lbl_path))

        # If mask is 3-channel, take the first channel
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Remap GOOSE class IDs to unified 7-class
        label = np.full_like(mask, 255, dtype=np.int64)
        if self._goose_id_to_unified:
            for goose_id, unified_id in self._goose_id_to_unified.items():
                label[mask == goose_id] = unified_id
        else:
            # Fallback: if no ID mapping provided, pass through
            # with a warning -- user must configure this
            label = mask.astype(np.int64)
            label[label >= NUM_CLASSES] = 255

        label = Image.fromarray(label.astype(np.uint8))
        return self._apply_transforms(image, label)


# ===================================================================
# Focal Loss
# ===================================================================

class FocalLoss(torch.nn.Module):
    """Focal Loss for severe class imbalance in off-road datasets.

    Down-weights well-classified (easy) pixels and focuses training
    on hard, often minority-class pixels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Important: alpha (class weight) must be applied SEPARATELY from
    the cross-entropy used to compute p_t. Passing weight= to
    F.cross_entropy and then computing pt = exp(-ce) gives
    pt = p_t^alpha instead of p_t, which distorts the focal
    modulation and can cause mode collapse.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs:  (B, C, H, W) raw logits
            targets: (B, H, W) ground truth class indices
        """
        # Step 1: Unweighted CE to get correct p_t
        # Do NOT pass weight= here; it would corrupt pt computation.
        ce_loss = F.cross_entropy(
            inputs, targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Step 2: p_t = probability of the true class
        pt = torch.exp(-ce_loss)

        # Step 3: Focal modulation — downweight easy (high pt) examples
        focal_term = (1.0 - pt) ** self.gamma

        # Step 4: Apply per-class alpha weight separately
        if self.alpha is not None:
            # Gather alpha for each pixel's true class
            # Clamp targets to valid range for indexing (ignore=255 -> 0)
            safe_targets = targets.clone()
            valid_mask = (targets != self.ignore_index)
            safe_targets[~valid_mask] = 0
            alpha_t = self.alpha[safe_targets]
            # Zero out ignored pixels
            alpha_t[~valid_mask] = 0.0
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        # Mean over valid (non-ignored) pixels only
        valid_mask = (targets != self.ignore_index)
        num_valid = valid_mask.sum().clamp(min=1)
        return focal_loss.sum() / num_valid


# ===================================================================
# EMA (Exponential Moving Average)
# ===================================================================

class EMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of parameters updated as a running average
    during training. EMA weights typically generalize better.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Call after each optimizer.step()."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model):
        """Replace model weights with EMA weights for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original training weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ===================================================================
# Dataset Builder
# ===================================================================

def build_dataset(data_config, is_train=True):
    """Build a combined dataset from all available off-road sources.

    Automatically discovers which datasets are available by checking
    if their root directories exist, and combines them via ConcatDataset.

    Args:
        data_config: dict with keys like rellis_root, rugd_root, goose_root,
                     rellis_split_train, rellis_split_val, crop_size
        is_train: bool

    Returns:
        torch.utils.data.Dataset
    """
    datasets = []
    crop_size = data_config.get("crop_size", 512)

    # RELLIS-3D
    rellis_root = data_config.get("rellis_root", "./data/Rellis-3D")
    if is_train:
        split_file = data_config.get("rellis_split_train",
                                      os.path.join(rellis_root, "split_custom/train_70.lst"))
    else:
        split_file = data_config.get("rellis_split_val",
                                      os.path.join(rellis_root, "split_custom/test_30.lst"))

    if os.path.exists(rellis_root) and os.path.exists(split_file):
        ds = Rellis3DUnified(rellis_root, split_file, is_train=is_train, crop_size=crop_size)
        if len(ds) > 0:
            datasets.append(ds)

    # RUGD (train only for now, or if split provided)
    rugd_root = data_config.get("rugd_root", "./data/RUGD")
    if os.path.isdir(rugd_root) and is_train:
        ds = RUGDUnified(rugd_root, is_train=is_train, crop_size=crop_size)
        if len(ds) > 0:
            datasets.append(ds)

    # GOOSE
    goose_root = data_config.get("goose_root", "./data/GOOSE")
    goose_id_map = data_config.get("goose_id_to_name", None)
    if os.path.isdir(goose_root) and is_train:
        split = "train" if is_train else "val"
        ds = GOOSEUnified(goose_root, split=split, is_train=is_train,
                          crop_size=crop_size, goose_id_to_name=goose_id_map)
        if len(ds) > 0:
            datasets.append(ds)

    if len(datasets) == 0:
        raise RuntimeError(
            "No training data found. Check dataset paths in config.\n"
            f"  rellis_root: {rellis_root}\n"
            f"  rugd_root:   {rugd_root}\n"
            f"  goose_root:  {goose_root}"
        )

    if len(datasets) == 1:
        return datasets[0]

    combined = ConcatDataset(datasets)
    print(f"[Combined] Total samples: {len(combined)} "
          f"({'+'.join(str(len(d)) for d in datasets)})")
    return combined