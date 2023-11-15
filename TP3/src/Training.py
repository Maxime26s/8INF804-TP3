import logging
from sklearn.metrics import cohen_kappa_score, classification_report
import torch
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, train_size):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader, desc="Training", unit="batch"):
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

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size

    return epoch_acc, epoch_loss


def validate_epoch(model, val_loader, criterion, device, val_size):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / val_size
    epoch_acc = running_corrects.double() / val_size

    return epoch_acc, epoch_loss


def test_model(model, test_loader, device, class_names):
    all_preds = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
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
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs} - Training Phase Start")
        train_epoch_acc, train_epoch_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, train_size
        )
        train_acc_history.append(train_epoch_acc.item())
        train_loss_history.append(train_epoch_loss)
        logging.info(
            f"Epoch {epoch}/{num_epochs} - Training Phase End - Accuracy: {train_epoch_acc:.4f} - Loss: {train_epoch_loss:.4f}"
        )

        logging.info(f"Epoch {epoch}/{num_epochs} - Validation Phase Start")
        val_epoch_acc, val_epoch_loss = validate_epoch(
            model, val_loader, criterion, device, val_size
        )
        val_acc_history.append(val_epoch_acc.item())
        val_loss_history.append(val_epoch_loss)
        logging.info(
            f"Epoch {epoch}/{num_epochs} - Validation Phase End - Accuracy: {val_epoch_acc:.4f} - Loss: {val_epoch_loss:.4f}"
        )

    logging.info("Testing Phase Start")
    kappa, classification_report_str = test_model(
        model, test_loader, device, class_names
    )
    logging.info("Testing Phase End")
    return (
        train_acc_history,
        val_acc_history,
        train_loss_history,
        val_loss_history,
        kappa,
        classification_report_str,
    )
