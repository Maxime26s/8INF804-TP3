from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import shutil


def is_flattened(directory):
    # Check if directory contains 'train' or 'val' subdirectories
    return not ("train" in os.listdir(directory) or "val" in os.listdir(directory))


def remove_unwanted_files(directory):
    for item in os.listdir(directory):
        if item.endswith(".txt") or item.endswith(".csv"):
            os.remove(os.path.join(directory, item))


def get_class_names_from_folders(*dirs):
    class_names = set()
    for directory in dirs:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                class_names.add(item)
    return class_names


def flatten_image_folder(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if is_flattened(data_dir):
        print("Directory is already flattened.")
        return

    remove_unwanted_files(data_dir)

    # Get all class names from the train and val directories
    class_names = get_class_names_from_folders(train_dir, val_dir)

    # Create class folders at the root and move images
    for class_name in class_names:
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        for subdir in [train_dir, val_dir]:
            remove_unwanted_files(subdir)
            source_folder = os.path.join(subdir, class_name)
            if os.path.exists(source_folder):
                for file in os.listdir(source_folder):
                    shutil.move(os.path.join(source_folder, file), class_folder)

    # Optional: Remove the now-empty train and val directories
    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)


def setup_data_loaders(data_dir, batch_size, num_workers):
    logging.info("Setting up data loaders")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    flatten_image_folder(data_dir)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=(num_workers // 2),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(num_workers // 4),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(num_workers // 4),
    )

    return train_loader, val_loader, test_loader


def check_label_distribution(loader):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())
    unique, counts = np.unique(all_labels, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    print("Label distribution:", label_distribution)


if __name__ == "__main__":
    batch_size = 64
    num_workers = 12
    train_loader, val_loader, test_loader = setup_data_loaders(
        "./images", batch_size, num_workers
    )

    # Check a single batch
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    print("Type of images:", images.dtype)
    print("Type of labels:", labels.dtype)

    check_label_distribution(train_loader)

    # visualize_sample_images(train_loader, class_names)
