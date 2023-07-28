
import os

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

WORKERS = os.cpu_count()

def create_dataloaders(
    train_path: Path,
    test_path: Path,
    transform: transforms.Compose,
    batch_size: int,
    workers: int = WORKERS
):
    """Creates train and test DataLoaders

    Creates DataLoaders to load data from train_path and test_path.

    Arguments:
        train_path: Path to training data directory.
        test_path: Path to testing data directory.
        transform: torchvision transform to apply over the training and testing data.
        batch_size: The number of samples in each batch of DataLoaders.
        workers: The number of workers per DataLoader to read the data.
    """
    # 1.  Create Datasets using datasets.ImageFolder
    train_data = datasets.ImageFolder(root=train_path, transform=transform)
    test_data = datasets.ImageFolder(root=test_path, transform=transform)

    # 2. Get the classes in the input data.
    class_names = train_data.classes
    class_dict = train_data.class_to_idx

    # 3. Print basic data information
    print('Training Set')
    print('------------')
    print(f'{train_data}')
    print(f'Classes: {train_data.classes}')
    print(f'Class Dictionary: {train_data.class_to_idx}')

    print('\nTest Set')
    print('------------')
    print(f'{test_data}')
    print(f'Classes: {test_data.classes}')
    print(f'Class Dictionary: {test_data.class_to_idx}')

    # 4. Batch the data using DataLoaders.
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
