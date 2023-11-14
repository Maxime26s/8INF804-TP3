import argparse
import matplotlib.pyplot as plt
from src.Utils import setup_data_loaders, initialize_logging, plot_learning_curves

plt.ion()
initialize_logging()


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

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Flatten the image folder if it isn't already
    flatten_image_folder(input_dir_path)

    # Set up data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(
        input_dir_path, batch_size, num_workers
    )

    # Get class names
    class_names = train_loader.dataset.dataset.classes

    # Initialize lists to store history for all runs
    all_train_acc_histories = []
    all_val_acc_histories = []
    all_kappa_scores = []

    for run in range(1, num_runs + 1):
        logging.info(f"Training run {run - 1}/{num_runs} started")

        # Load selected model
        if neural_network_to_use == "vgg16":
            model = initialize_vgg16_model(device)
        elif neural_network_to_use == "custom":
            model = initialize_custom_model(device)
        else:
            raise ValueError(
                f"Neural network {neural_network_to_use} not supported. Supported neural networks: vgg16, custom"
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

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

        logging.info(f"Training Run {run} - Training and validation completed")
        logging.info(f"Training Run {run} - Kappa Score: {kappa}")

        # Append the history of this run to the overall history
        all_train_acc_histories.append(train_acc_history)
        all_val_acc_histories.append(val_acc_history)
        all_kappa_scores.append(kappa)

        # Plot learning curves for this run
        plot_learning_curves(train_acc_history, val_acc_history, run)

        logging.info(
            f"Training Run {run} - Learning curves saved as 'learning_curves_run{run}.png'"
        )

    # After all runs, print the kappa scores for each run
    logging.info(f"All runs completed. Kappa scores for each run: {all_kappa_scores}")

    # Plot average learning curves
    plot_learning_curves(
        np.mean(all_train_acc_histories, axis=0),
        np.mean(all_val_acc_histories, axis=0),
        "avg",
    )
    logging.info(f"Average learning curves saved as 'learning_curves_avg.png'")
