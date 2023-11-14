import argparse

if __name__ == "__main__":
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

    args = arg_parser.parse_args()
    neural_network_to_use = args.neural_network.lower()
    input_dir_path = args.input
    output_dir_path = args.output
    num_epochs = args.epochs
    batch_size = args.batch_size
    num_runs = args.runs
    learning_rate = args.learning_rate
    num_workers = args.workers
