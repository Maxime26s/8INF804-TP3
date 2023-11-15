import argparse
import logging
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.Utils import (
    setup_data_loaders,
    initialize_logging,
    plot_learning_curves,
    flatten_image_folder,
)
from src.VGG16 import initialize_vgg16_model
from src.Training import train_and_evaluate_model


def parse_args():
    arg_parser = argparse.ArgumentParser(description="tp3")
    arg_parser.add_argument(
        "neural_network",
        help="neural network to use",
        type=str,
        choices=["vgg16", "custom"],
    )
    arg_parser.add_argument(
        "-i",
        "--input",
        help="input folder name",
        type=str,
        default="./images/",
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        help="output folder name",
        type=str,
        default="./output/",
    )
    arg_parser.add_argument(
        "-e",
        "--epochs",
        help="number of epochs to do",
        type=int,
        default=10,
    )
    arg_parser.add_argument(
        "-b",
        "--batch_size",
        help="batch size to use",
        type=int,
        default=64,
    )
    arg_parser.add_argument(
        "-r",
        "--runs",
        help="number of runs to do",
        type=int,
        default=5,
    )
    arg_parser.add_argument(
        "-l",
        "--learning_rate",
        help="learning rate to use",
        type=float,
        default=0.001,
    )
    arg_parser.add_argument(
        "-w",
        "--workers",
        help="number of workers to use",
        type=int,
        default=12,
    )
    arg_parser.add_argument(
        "-s",
        "--show",
        help="show learning curve graphs",
        action="store_true",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    neural_network_to_use = args.neural_network.lower()
    input_dir_path = args.input
    output_dir_path = args.output
    num_epochs = args.epochs
    batch_size = args.batch_size
    num_runs = args.runs
    learning_rate = args.learning_rate
    num_workers = args.workers
    should_show_images = args.show

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    initialize_logging(output_dir_path, neural_network_to_use)

    # Flatten the image folder if it isn't already
    flatten_image_folder(input_dir_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("CUDA Availability: " + str(torch.cuda.is_available()))
    logging.info("Using device: " + str(device))

    # Set up data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(
        input_dir_path, batch_size, num_workers
    )

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    # Get class names
    class_names = train_loader.dataset.dataset.classes

    runs_stats = pd.DataFrame(
        columns=[
            "Run",
            "Kappa Score",
            "Final Training Accuracy",
            "Final Training Loss",
            "Final Validation Accuracy",
            "Final Validation Loss",
            "Classification Report",
        ]
    )

    # Initialize lists to store history for all runs
    all_train_acc_histories = []
    all_val_acc_histories = []
    all_train_loss_histories = []
    all_val_loss_histories = []
    all_kappa_scores = []

    for run in range(1, num_runs + 1):
        logging.info(f"Training run {run}/{num_runs} started")

        # Load selected model
        if neural_network_to_use == "vgg16":
            model = initialize_vgg16_model(device)
        elif neural_network_to_use == "custom":
            # model = initialize_custom_model(device)
            pass
        else:
            raise ValueError(
                f"Neural network {neural_network_to_use} not supported. Supported neural networks: vgg16, custom"
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

        (
            train_acc_history,
            val_acc_history,
            train_loss_history,
            val_loss_history,
            kappa,
            classification_report_str,
        ) = train_and_evaluate_model(
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

        logging.info(f"Classification Report:\n{classification_report_str}")

        logging.info(f"Training Run {run} - Training and validation completed")
        logging.info(f"Training Run {run} - Kappa Score: {kappa}")

        new_row = {
            "Run": run,
            "Kappa Score": kappa,
            "Final Training Accuracy": train_acc_history[-1],
            "Final Training Loss": train_loss_history[-1],
            "Final Validation Accuracy": val_acc_history[-1],
            "Final Validation Loss": val_loss_history[-1],
            "Classification Report": classification_report_str,
        }
        runs_stats = pd.concat([runs_stats, pd.DataFrame([new_row])], ignore_index=True)

        # Append the history of this run to the overall history
        all_train_acc_histories.append(train_acc_history)
        all_val_acc_histories.append(val_acc_history)
        all_train_loss_histories.append(train_loss_history)
        all_val_loss_histories.append(val_loss_history)
        all_kappa_scores.append(kappa)

        # Plot learning curves for this run
        plot_learning_curves(
            train_acc_history,
            val_acc_history,
            train_loss_history,
            val_loss_history,
            run,
            neural_network_to_use,
            output_dir_path,
            should_show_images,
        )

        logging.info(
            f"Training Run {run} - Learning curves saved as 'learning_curves_run{run}.png'"
        )

    # After all runs, print the average kappa score
    logging.info(
        f"All runs completed. Average Kappa score: {np.mean(all_kappa_scores)}"
    )

    # Plot average learning curves
    plot_learning_curves(
        np.mean(all_train_acc_histories, axis=0),
        np.mean(all_val_acc_histories, axis=0),
        np.mean(all_train_loss_histories, axis=0),
        np.mean(all_val_loss_histories, axis=0),
        "avg",
        neural_network_to_use,
        output_dir_path,
        should_show_images,
    )
    logging.info(f"Average learning curves saved as 'learning_curves_avg.png'")

    csv_file_path = os.path.join(
        output_dir_path, f"{neural_network_to_use}_runs_stats.csv"
    )
    runs_stats.to_csv(csv_file_path, index=False)
    logging.info(f"Runs statistics saved as '{csv_file_path}'")
