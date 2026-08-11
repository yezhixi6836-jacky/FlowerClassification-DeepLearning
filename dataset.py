from torchvision.datasets import Flowers102

DATASET_ROOT = "../dataset"


def load_datasets(train_transform=None,
                  test_transform=None,
                  download=False):

    train_dataset = Flowers102(
        root=DATASET_ROOT,
        split="train",
        transform=train_transform,
        download=download
    )

    val_dataset = Flowers102(
        root=DATASET_ROOT,
        split="val",
        transform=test_transform,
        download=download
    )

    test_dataset = Flowers102(
        root=DATASET_ROOT,
        split="test",
        transform=test_transform,
        download=download
    )

    return train_dataset, val_dataset, test_dataset