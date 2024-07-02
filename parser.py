import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    
    parser.add_argument('--data_root', type=str, default='/home/juajua56/train', help='Root directory of the data.')
    parser.add_argument('--data_path', type=str, default='/home/juajua56/train_data.csv', help='Path to the data CSV file.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma value for the learning rate scheduler.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=9425, help='Random state for train-test split.')
    
    return parser.parse_args()
