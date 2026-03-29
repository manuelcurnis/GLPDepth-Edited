"""
Prepare ShapeNetSem dataset for GLPDepth finetuning.

Following the paper "Estimating Object Physical Properties from RGB-D Vision 
and Depth Robot Sensors Using Deep Learning" (Cardoso & Moreno, 2025):
- 8,948 models, 14 views each
- 90/10 train/test split by model (no model overlap)
- Depth normalized by bounding box diagonal
- Filter out models with zero-dimension bounding boxes

This script:
1. Reads metadata.csv to get aligned.dims per model
2. Filters models with valid (non-zero) dimensions that exist in output/
3. Splits 90/10 by model into train/test
4. Generates train.txt and list_test.txt file lists
5. Generates metadata.txt in the format expected by shapenetsem_normalized.py
6. Creates a symlink so datasets/shapenetsem points to the output folder
"""

import csv
import os
import random
import numpy as np

random.seed(42)  # Reproducibility

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, 'code', 'dataset', 'output')
METADATA_CSV = os.path.join(ROOT, 'code', 'dataset', 'metadata.csv')
FILENAMES_DIR = os.path.join(ROOT, 'code', 'dataset', 'filenames', 'shapenetsem')
DATASETS_DIR = os.path.join(ROOT, 'datasets')
SYMLINK_TARGET = os.path.join(DATASETS_DIR, 'shapenetsem')

# The 14 view suffixes (8 rotations + 6 cardinal)
VIEW_SUFFIXES = [
    'r_000', 'r_045', 'r_090', 'r_135', 'r_180', 'r_225', 'r_270', 'r_315',
    'back', 'bottom', 'front', 'left', 'right', 'top'
]


def parse_dims(dims_str):
    """Parse aligned.dims string like '111.97104\\,84.16881\\,0.0' into floats.
    Returns None if the string is empty, malformed, or doesn't have exactly 3 values."""
    if not dims_str or not dims_str.strip():
        return None
    try:
        parts = dims_str.replace('\\,', ',').split(',')
        values = [float(p.strip()) for p in parts if p.strip()]
        if len(values) != 3:
            return None
        return values
    except (ValueError, TypeError):
        return None


def main():
    # 1. Read metadata
    print("Reading metadata...")
    metadata = {}
    with open(METADATA_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row['fullId'].replace('wss.', '')
            aligned_dims = row.get('aligned.dims', '')
            if aligned_dims:
                metadata[model_id] = aligned_dims

    print(f"  Total metadata entries with dims: {len(metadata)}")

    # 2. Get output folders
    all_folders = sorted([f for f in os.listdir(OUTPUT_DIR) 
                          if os.path.isdir(os.path.join(OUTPUT_DIR, f)) and not f.startswith('.')])
    print(f"  Total output folders: {len(all_folders)}")

    # 3. Filter: must have metadata with non-zero dims AND all 14 views
    valid_models = []
    skipped_no_meta = 0
    skipped_bad_dims = 0
    skipped_missing_files = 0

    for model_id in all_folders:
        if model_id not in metadata:
            skipped_no_meta += 1
            continue

        dims = parse_dims(metadata[model_id])
        if dims is None or 0.0 in dims or any(d <= 0 for d in dims):
            skipped_bad_dims += 1
            continue

        # Check all 14 view pairs exist
        model_dir = os.path.join(OUTPUT_DIR, model_id)
        all_present = True
        for suffix in VIEW_SUFFIXES:
            png = os.path.join(model_dir, f"{model_id}_{suffix}.png")
            exr = os.path.join(model_dir, f"{model_id}_{suffix}_depth0001.exr")
            if not os.path.exists(png) or not os.path.exists(exr):
                all_present = False
                break
        
        if not all_present:
            skipped_missing_files += 1
            continue

        valid_models.append(model_id)

    print(f"\n  Valid models: {len(valid_models)}")
    print(f"  Skipped (no metadata): {skipped_no_meta}")
    print(f"  Skipped (empty/zero/bad dims): {skipped_bad_dims}")
    print(f"  Skipped (missing files): {skipped_missing_files}")

    # 4. Split 90/10 by model (as in the paper)
    random.shuffle(valid_models)
    split_idx = int(len(valid_models) * 0.9)
    train_models = sorted(valid_models[:split_idx])
    test_models = sorted(valid_models[split_idx:])

    print(f"\n  Train models: {len(train_models)} ({len(train_models) * 14} images)")
    print(f"  Test models:  {len(test_models)} ({len(test_models) * 14} images)")

    # 5. Create filenames directory
    os.makedirs(FILENAMES_DIR, exist_ok=True)

    # 6. Generate train.txt and list_test.txt
    # Format: /model_id/model_id_view.png /model_id/model_id_view_depth0001.exr
    def write_filelist(models, filename):
        filepath = os.path.join(FILENAMES_DIR, filename)
        lines = []
        for model_id in models:
            for suffix in VIEW_SUFFIXES:
                rgb_path = f"/{model_id}/{model_id}_{suffix}.png"
                depth_path = f"/{model_id}/{model_id}_{suffix}_depth0001.exr"
                lines.append(f"{rgb_path} {depth_path}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Written {filename}: {len(lines)} entries")
        return len(lines)

    print("\nGenerating file lists...")
    n_train = write_filelist(train_models, 'train.txt')
    n_test = write_filelist(test_models, 'list_test.txt')

    # 7. Generate metadata.txt (CSV with fullId and aligned.dims)
    # Format expected by shapenetsem_normalized.py: fullId,aligned.dims
    # Quote the dims field to prevent pandas from splitting on \,
    metadata_path = os.path.join(FILENAMES_DIR, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write('fullId,aligned.dims\n')
        for model_id in sorted(valid_models):
            dims_str = metadata[model_id]
            f.write(f'wss.{model_id},"{dims_str}"\n')
    print(f"  Written metadata.txt: {len(valid_models)} entries")

    # 8. Create datasets/shapenetsem symlink -> code/dataset/output
    os.makedirs(DATASETS_DIR, exist_ok=True)
    if os.path.exists(SYMLINK_TARGET):
        if os.path.islink(SYMLINK_TARGET):
            os.unlink(SYMLINK_TARGET)
            print(f"\n  Removed old symlink: {SYMLINK_TARGET}")
        else:
            print(f"\n  WARNING: {SYMLINK_TARGET} already exists and is not a symlink!")
            print(f"  Please remove it manually and re-run.")
            return
    
    os.symlink(OUTPUT_DIR, SYMLINK_TARGET)
    print(f"  Created symlink: {SYMLINK_TARGET} -> {OUTPUT_DIR}")

    # 9. Verify a sample
    print("\n--- Verification ---")
    sample_model = train_models[0]
    dims = parse_dims(metadata[sample_model])
    diag = np.linalg.norm(np.array(dims) / 100.0)  # Convert to meters
    print(f"  Sample model: {sample_model}")
    print(f"  Dims (cm): {dims}")
    print(f"  Diagonal (m): {diag:.4f}")
    print(f"  Max normalized depth (paper uses 2.63): ~{diag:.2f}")

    # Compute max normalization across all models for max_depth setting
    all_diags = []
    for model_id in valid_models:
        dims = parse_dims(metadata[model_id])
        dims_m = np.array(dims) / 100.0
        all_diags.append(np.linalg.norm(dims_m))
    
    print(f"\n  Diagonal stats (meters):")
    print(f"    Min:  {min(all_diags):.4f}")
    print(f"    Max:  {max(all_diags):.4f}")
    print(f"    Mean: {np.mean(all_diags):.4f}")
    print(f"    The paper uses max_depth=2.63")

    print(f"\n✅ Dataset preparation complete!")
    print(f"\nTo finetune, run:")
    print(f"  python ./code/train.py \\")
    print(f"    --dataset shapenetsem_normalized \\")
    print(f"    --data_path ./datasets/ \\")
    print(f"    --max_depth 2.63 \\")
    print(f"    --max_depth_eval 2.63 \\")
    print(f"    --ckpt_dir ./code/models/best_model_nyu.ckpt \\")
    print(f"    --save_model \\")
    print(f"    --epochs 25")


if __name__ == '__main__':
    main()
