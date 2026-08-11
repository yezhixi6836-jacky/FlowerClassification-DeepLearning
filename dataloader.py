from torch.utils.data import DataLoader
from utils.dataset import load_datasets
from utils.transforms import train_transform, test_transform

def get_dataloaders(batch_size=32, num_workers=0):
    train_dataset, val_dataset, test_dataset = load_datasets(
        train_transform=train_transform,
        test_transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader