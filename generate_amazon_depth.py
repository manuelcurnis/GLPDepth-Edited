"""
Batch inference script: run finetuned GLPDepth on all Amazon splits
and create a dataset folder compatible with Depth-mass-estimator
(https://github.com/RavineWindteer/Depth-mass-estimator).

Expected layout by Depth-mass-estimator:
    datasets/amazon/
        train/
            images/  (.pklz files)
            depth/   (.png depth maps, same stem as .pklz)
        test/
            images/  (.pklz files or amazon_test_set.pklz)
            depth/   (.png depth maps, named {idx}.png for test)
        val/
            images/  (.pklz files)
            depth/   (.png depth maps, same stem as .pklz)

    code/dataset/filenames/amazon/
        file_paths_train.txt   (lines like: train/100000_Tool Boxes nice.pklz)
        file_paths_test.txt    (lines like: test/6530_ignition coil large.pklz)
        file_paths_val.txt     (lines like: val/1_hard drive sturdy.pklz)

Source data is in: datasets/amazon_data/{train,test,val}_data/*.pklz

Usage:
    python generate_amazon_depth.py --gpu_or_cpu cpu --splits train test val
    python generate_amazon_depth.py --gpu_or_cpu cpu --splits test
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pickle
import lz4.block
from io import BytesIO
from PIL import Image
from collections import OrderedDict

import torch
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
from models.model import GLPDepth


# ---------- constants ----------
MAX_DEPTH = 2.63
CKPT_PATH = './code/models/best_model_cosine_adamw.ckpt'
SRC_DIR = './datasets/amazon_data'
DST_DIR = './datasets/amazon'
SPLITS = ['train', 'test', 'val']

# Map our split names to source folder names
SRC_SPLIT_MAP = {
    'train': 'train_data',
    'test':  'test_data',
    'val':   'val_data',
}

to_tensor = transforms.ToTensor()


# ---------- helpers (same logic as inference.py / notebook) ----------

def load_pklz(filepath):
    with open(filepath, 'rb') as fp:
        compressed_bytes = fp.read()
    decompressed = lz4.block.decompress(compressed_bytes)
    return pickle.loads(decompressed)


def resize_and_pad(image):
    border_size = (640 - 480) // 2
    image = cv2.resize(image, (480, 480))
    image = cv2.copyMakeBorder(image, 0, 0, border_size, border_size,
                               cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return image


def get_normalization(dimensions_str):
    dims = np.array([float(d) for d in dimensions_str])
    dims_meters = dims * 2.54 / 100.0
    return np.linalg.norm(dims_meters)


def crop_prediction(pred, input_image, device):
    """Mask background (white) pixels and crop to eval region."""
    pred[torch.isinf(pred)] = MAX_DEPTH
    pred[torch.isnan(pred)] = 0.0

    gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

    valid_mask = torch.from_numpy(bin_img > (255 / 2)).to(device=device)
    eval_mask = torch.zeros(valid_mask.shape, device=device)
    eval_mask[45:471, 41:601] = 1
    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return torch.where(valid_mask, pred, torch.zeros_like(pred))


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description='Generate depth maps for Amazon dataset')
    parser.add_argument('--gpu_or_cpu', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--ckpt_dir', type=str, default=CKPT_PATH)
    parser.add_argument('--src_dir', type=str, default=SRC_DIR)
    parser.add_argument('--dst_dir', type=str, default=DST_DIR)
    parser.add_argument('--splits', type=str, nargs='+', default=SPLITS,
                        help='which splits to process (train, test, val)')
    parser.add_argument('--compression', type=int, default=3,
                        help='PNG compression level 0-9 (higher=smaller but slower)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu_or_cpu == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt_dir}")
    model = GLPDepth(max_depth=MAX_DEPTH, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir, map_location=device)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    # Process each split
    for split in args.splits:
        src_folder = SRC_SPLIT_MAP.get(split)
        if not src_folder:
            print(f"Unknown split: {split}")
            continue

        src_split = os.path.join(args.src_dir, src_folder)
        dst_images = os.path.join(args.dst_dir, split, 'images')
        dst_depth = os.path.join(args.dst_dir, split, 'depth')

        if not os.path.isdir(src_split):
            print(f"Skipping {split}: {src_split} not found")
            continue

        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_depth, exist_ok=True)

        pklz_files = sorted([f for f in os.listdir(src_split) if f.endswith('.pklz')])
        total = len(pklz_files)
        print(f"\n=== {split}: {total} files ===")
        print(f"  images → {dst_images}")
        print(f"  depth  → {dst_depth}")

        # Build file paths list for Depth-mass-estimator
        file_paths = []

        skipped = 0
        for i, fname in enumerate(pklz_files):
            # Depth filename: same stem as .pklz, just .png extension
            depth_filename = fname.replace('.pklz', '.png')
            dst_depth_path = os.path.join(dst_depth, depth_filename)

            # Symlink .pklz into images/ (avoid duplicating ~3GB of data)
            src_pklz = os.path.join(src_split, fname)
            dst_pklz = os.path.join(dst_images, fname)
            if not os.path.lexists(dst_pklz):
                os.symlink(os.path.abspath(src_pklz), dst_pklz)

            # Record file path for txt file
            file_paths.append(f"{split}/{fname}")

            # Skip depth if already generated (allows resuming)
            if os.path.exists(dst_depth_path):
                skipped += 1
                continue

            try:
                record = load_pklz(src_pklz)
                raw_image = np.array(Image.open(BytesIO(record['image_data'])))
                dimensions = record['dimensions']

                image = resize_and_pad(raw_image)
                normalization = get_normalization(dimensions)

                image_tensor = to_tensor(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(image_tensor)

                pred_d = pred['pred_d'].squeeze() * normalization
                pred_d_mm = pred_d * 1000.0
                pred_d_mm = crop_prediction(pred_d_mm, image, device)
                pred_d_np = pred_d_mm.cpu().numpy()

                cv2.imwrite(dst_depth_path, pred_d_np.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, args.compression])

            except Exception as e:
                print(f"\n  ERROR on {fname}: {e}")
                continue

            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] processed ({skipped} skipped)")

        print(f"  Done: {total - skipped} new, {skipped} skipped")

        # Write file_paths txt for Depth-mass-estimator
        filenames_dir = os.path.join('./code/dataset/filenames/amazon')
        os.makedirs(filenames_dir, exist_ok=True)
        txt_path = os.path.join(filenames_dir, f'file_paths_{split}.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(file_paths) + '\n')
        print(f"  Wrote {txt_path} ({len(file_paths)} entries)")

    print(f"\nAll done. Dataset saved to: {args.dst_dir}")


if __name__ == '__main__':
    main()
