import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import cohen_kappa_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


def setup_data_loaders(batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_dir = "./images"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# Function to train and evaluate the model
def train_and_evaluate_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    train_size,
    val_size,
    class_names,
):
    best_kappa = 0.0
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate training accuracy
        epoch_acc = running_corrects.double() / train_size
        train_acc_history.append(epoch_acc.item())

        # Validation loop
        model.eval()
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_acc = val_running_corrects.double() / val_size
        val_acc_history.append(val_epoch_acc.item())

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f}"
        )

    # Test loop for Kappa and classification report
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())

    kappa = cohen_kappa_score(all_labels, all_preds)
    if kappa > best_kappa:
        best_kappa = kappa
        print(f"New best Kappa score: {best_kappa:.4f}")
        print(classification_report(all_labels, all_preds, target_names=class_names))

    return train_acc_history, val_acc_history, best_kappa


def main():
    print(torch.cuda.is_available())
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 64
    num_workers = 4

    # Data loaders for each set
    train_loader, val_loader, test_loader = setup_data_loaders(batch_size, num_workers)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    class_names = (
        train_loader.dataset.dataset.classes
    )  # Assuming class names are the same for all sets

    # Initialize lists to store history for all runs
    all_train_acc_histories = []
    all_val_acc_histories = []
    all_kappa_scores = []

    num_runs = 5
    for run in range(num_runs):
        print(f"Training run {run + 1}/{num_runs}")

        # Load VGG16 pre-trained model
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights).to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 14)  # 14 classes for flower types
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

        num_epochs = 10
        train_acc_history, val_acc_history, best_kappa = train_and_evaluate_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
            train_size,
            val_size,
            class_names,
        )

        # Append the history of this run to the overall history
        all_train_acc_histories.append(train_acc_history)
        all_val_acc_histories.append(val_acc_history)
        all_kappa_scores.append(best_kappa)

        # Plot learning curves for this run
        plt.plot(train_acc_history, label="Train Accuracy")
        plt.plot(val_acc_history, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curves (Run {run + 1})")
        plt.legend()
        plt.savefig(f"transfer_run{run + 1}.png")
        plt.show()

    # After all runs, print the kappa scores for each run
    print("Kappa scores for each run:", all_kappa_scores)


if __name__ == "__main__":
    main()
