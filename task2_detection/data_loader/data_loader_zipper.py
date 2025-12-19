import os
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ClassDataset(Dataset):
    """Dataset that loads only zipper images (good=0, bad=1)."""

    def __init__(self, root: str, split: str = "train", img_size: int = 64):
        self.root = Path(root)
        self.split = split

        self.samples = []   # [(path, label, domain), ...]

        # ✔ 只加载 zipper（domain = 1）
        for domain_id, category in [(1, "zipper")]:
            base = self.root / category / split

            good_dir = base / "good"
            bad_dir  = base / "bad"

            # good = label 0
            if good_dir.exists():
                for f in sorted(good_dir.iterdir()):
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.samples.append((f, 0, domain_id))

            # bad = label 1
            if bad_dir.exists():
                for f in sorted(bad_dir.iterdir()):
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.samples.append((f, 1, domain_id))

        # if split == "train":
        #     self.transform = transforms.Compose([
        #         transforms.Resize((img_size, img_size)),
        #         transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomRotation(12),
        #         transforms.ColorJitter(
        #             brightness=0.3,
        #             contrast=0.3,
        #             saturation=0.2,
        #             hue=0.02
        #         ),
        #         transforms.ToTensor(),
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.Resize((img_size, img_size)),
        #         transforms.ToTensor(),
        #     ])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label), torch.tensor(domain)



def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 64,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, Dict]:

    train_dataset = ClassDataset(dataset_dir, split="train", img_size=img_size)
    test_dataset  = ClassDataset(dataset_dir, split="test", img_size=img_size)

    class_to_idx = {"good":0, "bad":1}  # clear and explicit

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, class_to_idx


if __name__ == "__main__":
    train_loader, test_loader, class_to_idx = create_dataloaders("dataset")

    from collections import Counter
    cnt_label = Counter()
    cnt_domain = Counter()
    for _, labels, domains in train_loader:
        cnt_label.update(labels.tolist())
        cnt_domain.update(domains.tolist())
    print(f"Labels: {cnt_label}")
    print(f"Domains: {cnt_domain}")

