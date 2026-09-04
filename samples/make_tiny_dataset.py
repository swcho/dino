"""Build a tiny ImageFolder-shaped dataset (train/ + val/) for DINO smoke tests.

Two sources:
  synth  - procedurally generated color-blob classes, no download, instant.
  cifar  - a per-class subset of CIFAR-10 upscaled to 96x96 (downloads ~170MB once).

Usage:
    python samples/make_tiny_dataset.py --out /tmp/dino_tiny --source synth
    python samples/make_tiny_dataset.py --out /tmp/dino_cifar --source cifar
"""
import argparse
import os

import numpy as np
from PIL import Image


def build_synth(out, n_train, n_val, size):
    classes = "abcde"
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        for ci, c in enumerate(classes):
            d = os.path.join(out, split, c)
            os.makedirs(d, exist_ok=True)
            base = np.array([40 + ci * 45, 200 - ci * 35, 60 + ci * 30])
            for i in range(n):
                arr = (base + rng.normal(0, 20, (size, size, 3))).clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(os.path.join(d, f"{i:03d}.png"))
    print(f"synth: {len(classes) * n_train} train / {len(classes) * n_val} val, {len(classes)} classes -> {out}")


def build_cifar(out, n_train, n_val, size):
    from torchvision.datasets import CIFAR10

    cache = os.path.join(out, "_cache")
    for split, per_class in (("train", n_train), ("val", n_val)):
        ds = CIFAR10(cache, train=(split == "train"), download=True)
        counts = {}
        for img, lab in zip(ds.data, ds.targets):
            c = ds.classes[lab]
            if counts.get(c, 0) >= per_class:
                continue
            counts[c] = counts.get(c, 0) + 1
            d = os.path.join(out, split, c)
            os.makedirs(d, exist_ok=True)
            Image.fromarray(img).resize((size, size), Image.BICUBIC).save(
                os.path.join(d, f"{counts[c]:04d}.png"))
        print(f"cifar {split}: {sum(counts.values())} images, {len(counts)} classes")
    print(f"-> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Tiny ImageFolder dataset for DINO smoke tests")
    p.add_argument("--out", required=True, help="Output root; train/ and val/ are created under it.")
    p.add_argument("--source", default="synth", choices=["synth", "cifar"])
    # eval_knn.py chunks the val set by len(val)//100, so keep val >= 100 images.
    p.add_argument("--n_train", default=60, type=int, help="Images per class in train/.")
    p.add_argument("--n_val", default=40, type=int, help="Images per class in val/.")
    p.add_argument("--size", default=160, type=int, help="Square image side in pixels.")
    args = p.parse_args()

    if args.source == "synth":
        build_synth(args.out, args.n_train, args.n_val, args.size)
    else:
        build_cifar(args.out, args.n_train, args.n_val, min(args.size, 96))
