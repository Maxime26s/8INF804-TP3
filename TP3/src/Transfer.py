import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import logging

from Utils import setup_data_loaders, initialize_logging, plot_learning_curves


def initialize_vgg16_model(device):
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 14)  # 14 classes
    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device, train_size):
    model.train()
    running_loss = 0.0
    running_corrects = 0

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

    return running_corrects.double() / train_size


def validate_epoch(model, val_loader, device, val_size):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    return running_corrects.double() / val_size


def test_model(model, test_loader, device, class_names):
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
    classification_report_str = classification_report(
        all_labels, all_preds, target_names=class_names
    )

    return kappa, classification_report_str


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
    logging.info("Starting training and evaluation")
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Training Phase Start")
        train_epoch_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, train_size
        )
        train_acc_history.append(train_epoch_acc.item())
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} - Training Phase End - Accuracy: {epoch_acc:.4f}"
        )

        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Validation Phase Start")
        val_epoch_acc = validate_epoch(model, val_loader, device, val_size)
        val_acc_history.append(val_epoch_acc.item())
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} - Validation Phase End - Accuracy: {val_epoch_acc:.4f}"
        )

    logging.info("Starting Kappa Score Evaluation and Classification Report Generation")
    kappa, classification_report_str = test_model(
        model, test_loader, device, class_names
    )
    logging.info(f"Classification Report:\n{classification_report_str}")
    return train_acc_history, val_acc_history, kappa


def main():
    plt.ion()  # interactive mode

    initialize_logging()

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("CUDA Availability: " + str(torch.cuda.is_available()))
    logging.info("Using device: " + str(device))

    # Data loaders for each set
    data_dir = "./images"
    batch_size = 64
    num_workers = 12
    train_loader, val_loader, test_loader = setup_data_loaders(
        data_dir, batch_size, num_workers
    )
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
        logging.info(f"Training run {run + 1}/{num_runs} started")

        # Load VGG16 pre-trained model
        logging.info("Loading VGG16 pre-trained model")
        model = initialize_vgg16_model(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

        num_epochs = 10
        train_acc_history, val_acc_history, kappa = train_and_evaluate_model(
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

        logging.info(f"Training Run {run + 1} - Training and validation completed")
        logging.info(f"Training Run {run + 1} - Kappa Score: {kappa}")

        # Append the history of this run to the overall history
        all_train_acc_histories.append(train_acc_history)
        all_val_acc_histories.append(val_acc_history)
        all_kappa_scores.append(kappa)

        # Plot learning curves for this run
        plot_learning_curves(train_acc_history, val_acc_history, run)

        logging.info(
            f"Training Run {run + 1} - Learning curves saved as 'learning_curves_run{run + 1}.png'"
        )

    # After all runs, print the kappa scores for each run
    logging.info(f"All runs completed. Kappa scores for each run: {all_kappa_scores}")


if __name__ == "__main__":
    main()
