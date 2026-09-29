"""
Download CIFAR-10 via torchvision (fast, works from China).
"""
from torchvision.datasets import CIFAR10
from pathlib import Path

out = Path("data/cifar10")
out.mkdir(parents=True, exist_ok=True)

print("Downloading CIFAR-10 train split...")
train = CIFAR10(root=out, train=True, download=True)
print(f"  Train images: {len(train)}")

print("Downloading CIFAR-10 test split...")
test = CIFAR10(root=out, train=False, download=True)
print(f"  Test images:  {len(test)}")

print(f"\nCIFAR-10 classes ({len(train.classes)}): {train.classes}")
print(f"Saved to {out}")
