import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a recommendation model.")

    # Data arguments
    parser.add_argument('--data_root', type=str, default='/home/juajua56/train', help='Root directory of the data.')
    parser.add_argument('--data_path', type=str, default='/home/juajua56/train_data.csv', help='Path to the data CSV file.')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')

    # Model hyperparameters
    parser.add_argument('--input_dim', type=int, default=768, help='Embedding dimension.')
    parser.add_argument('--output_dim', type=int, default=24, help='Number of classes.')
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout ratio.')

    # Optimizer and scheduler parameters
    parser.add_argument('--step_size', type=int, default=5, help='Step size for the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma value for the learning rate scheduler.')

    # Reproducibility
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')

    # Train-test split
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=9425, help='Random state for train-test split.')

    # GPU settings
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0" for GPU 0).')

    return parser.parse_args()
