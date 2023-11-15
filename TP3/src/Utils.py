import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


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


def initialize_logging(output_dir_path, neural_network):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"{output_dir_path}/{neural_network}_training.log",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def plot_learning_curves(
    train_acc_history,
    val_acc_history,
    train_loss_history,
    val_loss_history,
    run,
    neural_network,
    output_dir_path,
    should_show,
):
    # Plotting acc
    plt.figure()
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curves (Run {run})")
    plt.legend()
    plt.savefig(f"{output_dir_path}/{neural_network}_accuracy_curves_run{run}.png")
    if should_show:
        plt.show()

    # Plotting loss
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves (Run {run})")
    plt.legend()
    plt.savefig(f"{output_dir_path}/{neural_network}_loss_curves_run{run}.png")


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
