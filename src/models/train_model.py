import argparse
import logging
import os
from typing import Dict, Literal

import torch

from ..data import DatasetLoader
from ..distance_metrics import DTW, EMD, L1, L2, LCSS, DistanceMetric
from .knn import kNNClassifier
from .lstm import LSTMClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Map metric parameter to corresponding distance metric class
DISTANCE_METRICS: Dict[str, DistanceMetric] = {
    "l1": L1,
    "l2": L2,
    "dtw": DTW,
    "lcss": LCSS,
    "emd": EMD,
}


class ClassifierTrainer:
    def __init__(
        self,
        model_type: Literal["knn", "lstm"],
        dataset_name: str,
        feature_type: Literal["joints", "angles", "fourier"],
    ):
        """Initializes the ClassifierTrainer.

        Args:
            model_type (str): Type of classifier ("lstm" or "knn").
            dataset_name (str): Name of the dataset.
            feature_type (str): Type of feature to use ("joints", "angles", "fourier").
        """
        self.model_type: Literal["knn", "lstm"] = model_type
        self.dataset_name: str = dataset_name
        self.feature_type: Literal["joints", "angles", "fourier"] = feature_type
        self.loader: DatasetLoader = DatasetLoader()
        self.X, self.y = self.loader.load_data_and_labels(dataset_name, feature_type)

    def train_lstm(
        self,
        hidden_size: int,
        num_layers: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        """Trains an LSTM classifier.

        Args:
            hidden_size (int): Hidden state size.
            num_layers (int): Number of LSTM layers.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimizer.
        """
        lstm = LSTMClassifier(
            feature_type=self.feature_type,
            num_classes=self.loader.num_classes,
            input_size=self.loader.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        lstm = lstm.fit(self.X, self.y)

        # Save the trained LSTM model
        model_filename = f"{self.dataset_name}_lstm_{self.feature_type}_model.pth"
        model_path = os.path.join("models/", model_filename)
        torch.save(lstm.model.state_dict(), model_path)
        logging.info(f"Trained LSTM model saved to: {model_path}")

    def train_knn(
        self,
        k: int,
        metric: Literal["l1", "l2", "dtw", "lcss", "emd"],
        weights: Literal["uniform", "distance"],
    ):
        """Trains a k-Nearest Neighbors (kNN) classifier.

        Args:
            k (int): Number of neighbors.
            metric (str): Distance metric for kNN.
            weights (str): Weight function for kNN ("uniform" or "distance").
        """
        chosen_metric = DISTANCE_METRICS.get(metric)
        if chosen_metric is None:
            raise ValueError("Invalid metric. Choose from 'l1', 'l2', 'dtw', 'lcss', 'emd'.")

        knn = kNNClassifier(
            feature_type=self.feature_type,
            k=k,
            metric=chosen_metric,
            metric_params=None,  # ! metric params to integrate
            weights=weights,
        )
        knn = knn.fit(self.X, self.y)

        # Save the trained kNN model
        model_filename = f"{self.dataset_name}_knn_{self.feature_type}_model.pth"
        model_path = os.path.join("models/", model_filename)
        torch.save(knn, model_path)
        logging.info(f"Trained kNN model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM or kNN classifier")

    # Common arguments for both kNN and LSTM
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lstm", "knn"],
        help="Type of classifier (LSTM or kNN)",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        choices=["joints", "angles", "fourier"],
        help="Type of feature to use",
    )

    # Arguments only for kNN
    parser.add_argument("--k", type=int, help="Number of neighbors (kNN only)")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["l1", "l2", "dtw", "lcss", "emd"],
        help="Distance metric for kNN (kNN only)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=["uniform", "distance"],
        help="Weight function for kNN (kNN only)",
    )

    # Arguments only for LSTM
    parser.add_argument("--hidden_size", type=int, help="Hidden state size (LSTM only)")
    parser.add_argument("--num_layers", type=int, help="Number of LSTM layers (LSTM only)")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs (LSTM only)")
    parser.add_argument("--batch_size", type=int, help="Batch size for training (LSTM only)")
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for optimizer (LSTM only)"
    )

    args = parser.parse_args()

    # Validate feature_type and metric conditions
    if args.model_type == "knn":
        if args.feature_type in ["joints", "angles"]:
            valid_metrics = ["l1", "l2", "dtw", "lcss"]
        elif args.feature_type == "fourier":
            valid_metrics = ["l1", "l2", "emd"]
        else:
            raise ValueError("Invalid feature_type. Choose from 'joints', 'angles', 'fourier'.")

        if args.metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric for feature_type='{args.feature_type}'. Choose from {valid_metrics}."
            )

    trainer = ClassifierTrainer(args.model_type, args.dataset, args.feature_type)

    if args.model_type == "lstm":
        trainer.train_lstm(
            hidden_size=args.hidden_size or trainer.loader.hidden_size,
            num_layers=args.num_layers or trainer.loader.num_layers,
            num_epochs=args.num_epochs or trainer.loader.num_epochs,
            batch_size=args.batch_size or trainer.loader.batch_size,
            learning_rate=args.learning_rate or trainer.loader.learning_rate,
        )
    elif args.model_type == "knn":
        trainer.train_knn(
            k=args.k or trainer.loader.k,
            metric=args.metric,
            weights=args.weights,
        )
    else:
        raise ValueError("Invalid model_type. Choose 'lstm' or 'knn'.")


if __name__ == "__main__":
    main()
