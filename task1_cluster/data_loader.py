"""Task 1 data loader for the clustering dataset."""
 
from __future__ import annotations
 
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
 
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
 
 
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
 

@dataclass(frozen=True)
class Sample:
    """Represents one labeled image sample."""

    path: Path
    label: int


class ClusterDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Basic dataset that returns image tensor + label."""

    def __init__(self, samples: Sequence[Sample], transform: Callable | None = None) -> None:
        self.samples = list(samples)
        self.transform = transform or self._default_transform()

    @staticmethod
    def _default_transform() -> Callable:
        if transforms is None:
            raise ImportError(
                "torchvision is required for the default transforms. "
                "Please provide `transform` explicitly or install torchvision."
            )
        return transforms.Compose(
            [
                transforms.Resize(DEFAULT_IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, sample.label


def _load_labels(labels_path: Path) -> Dict[str, str]:
    with labels_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_samples(dataset_dir: Path, labels_map: Dict[str, str]) -> Tuple[List[Sample], Dict[str, int]]:
    class_to_idx: Dict[str, int] = {}
    samples: List[Sample] = []

    for file_name, class_name in labels_map.items():
        image_path = dataset_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_path} missing for label entry.")

        label_idx = class_to_idx.setdefault(class_name, len(class_to_idx))
        samples.append(Sample(image_path, label_idx))

    return samples, class_to_idx


def _split_samples(samples: Sequence[Sample], test_size: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    labels = [s.label for s in samples]
    train_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return list(train_samples), list(test_samples)


def create_datasets(
    dataset_dir: str | Path = "datasets/dataset",
    labels_path: str | Path = "datasets/cluster_labels.json",
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_RANDOM_STATE,
    transform: Callable | None = None,
) -> Tuple[ClusterDataset, ClusterDataset, Dict[str, int]]:
    """
    Build train/test datasets along with class-id mapping.

    Returns (train_dataset, test_dataset, class_to_idx).
    """

    dataset_dir = Path(dataset_dir)
    labels_path = Path(labels_path)

    labels_map = _load_labels(labels_path)
    samples, class_to_idx = _build_samples(dataset_dir, labels_map)
    train_samples, test_samples = _split_samples(samples, test_size, seed)

    train_dataset = ClusterDataset(train_samples, transform=transform)
    test_dataset = ClusterDataset(test_samples, transform=transform)

    return train_dataset, test_dataset, class_to_idx


def create_dataloaders(batch_size: int = 32, num_workers: int = 0, shuffle_train: bool = True, **dataset_kwargs) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Convenience helper that wraps datasets with DataLoaders."""

    train_dataset, test_dataset, class_to_idx = create_datasets(**dataset_kwargs)

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


__all__ = [
    "ClusterDataset",
    "create_datasets",
    "create_dataloaders",
    "Sample",
]

