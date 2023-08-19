import os
from typing import List, Literal, Tuple, Union

import pandas as pd

from ..features import AngleSeries, FourierSeries, JointSeries

DATASET_FOLDER = "dataset"
PROCESSED_SUBFOLDER = "processed"


class DatasetLoader:
    """A class for loading data and labels from processed dataset folders."""

    def __init__(self):
        self.dataset_folder = os.path.join(DATASET_FOLDER, PROCESSED_SUBFOLDER)
        self.num_classes: int = None
        self.input_size: int = None

    def load_data_and_labels(
        self, dataset_name: str, feature_type: Literal["joints", "angles", "fourier"]
    ) -> Tuple[List[Union[JointSeries, AngleSeries, FourierSeries]], List[str]]:
        """Load data and labels from processed dataset folders.

        Args:
            dataset_name (str): Name of the dataset.
            feature_type (Literal["joints", "angles", "fourier"]): Type of feature to load.

        Returns:
            Tuple[List[Union[JointSeries, AngleSeries, FourierSeries]], List[str]]: Loaded data and corresponding labels.
        """
        dataset_path = os.path.join(self.dataset_folder, dataset_name)

        X = []
        y = []

        for label in os.listdir(dataset_path):
            label_folder = os.path.join(dataset_path, label, feature_type)
            if os.path.exists(label_folder):
                for file_name in os.listdir(label_folder):
                    if file_name.endswith(".pkl"):
                        file_path = os.path.join(label_folder, file_name)
                        x = pd.read_pickle(file_path)
                        X.append(x)
                        y.append(label)

        self.num_classes = len(set(y))
        self.input_size = len(X[0].columns)

        return X, y
