import logging
import operator
from collections import Counter
from typing import Dict, List, Literal, Tuple, Union

from ..distance_metrics.distance_metric import DistanceMetric
from ..features import AngleSeries, FourierSeries, JointSeries
from .model import ClassifierModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class kNNClassifier(ClassifierModel):
    def __init__(
        self,
        feature_type: Literal["joints", "angles", "fourier"],
        k: int,
        metric: DistanceMetric,
        metric_params: dict = None,
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        """Initialize the kNNClassifier.

        Args:
            k (int): Number of nearest neighbors to consider.
            metric (Callable): Distance metric function to compute distances between data points.
            weights (Literal["uniform", "distance"]): Weighting method for predictions.
        """
        logging.info(f"Initializing kNN classifier ({k=}, metric={metric.name}, {weights=})")
        self.k: int = k
        self.train_data: List[Union[JointSeries, AngleSeries, FourierSeries]] = None
        self.train_labels: List[str] = None
        self.all_labels: Dict[str] = {}
        self.metric: DistanceMetric = metric(**metric_params) if metric_params else metric()
        self.weights: Literal["uniform", "distance"] = weights

    def fit(self, X: List[Union[JointSeries, AngleSeries, FourierSeries]], y: List[str]) -> None:
        """Fit the kNN classifier with training data and labels.

        Args:
            X (List): List of input data.
            y (List): List of corresponding labels.
        """
        self.train_data = X
        self.train_labels = y
        self.all_labels = set(y)
        logging.info(
            f"Fitting kNN classifier to training data: {len(self.train_data)} sample(s), {len(self.all_labels)} unique labels"
        )
        return self

    def _get_neighbors(
        self, _x: Union[JointSeries, AngleSeries, FourierSeries]
    ) -> List[Tuple[str, int]]:
        """Get the k nearest neighbors and their distances for a given data point.

        Args:
            _x (Union[JointSeries, AngleSeries, FourierSeries]): Input data point.

        Returns:
            List[Tuple[str, int]]: List of (label, distance) tuples for nearest neighbors.
        """
        distances_and_labels = [
            (neighbor_label, self.metric.compute_distance(series_1=_x, series_2=neighbor))
            for neighbor, neighbor_label in zip(self.train_data, self.train_labels)
        ]
        distances_and_labels.sort(key=operator.itemgetter(1))
        return distances_and_labels[: self.k]

    def predict(
        self,
        X: Union[
            Union[JointSeries, AngleSeries, FourierSeries],
            List[Union[JointSeries, AngleSeries, FourierSeries]],
        ],
    ) -> Union[List[str], List[Dict[str, float]]]:
        """Predict class labels for input data points.

        Args:
            X (Union[Union[JointSeries, AngleSeries, FourierSeries], List]):
                Input data point or list of input data points.

        Returns:
            Union[List[str], List[Dict[str, float]]]:
                List of predicted class labels or list of dictionaries containing class probabilities.
        """
        if not isinstance(X, list):
            single_input = True
            X = [X]
        else:
            single_input = False
        logging.info(f"Predicting class labels for input data: {len(X)} sample(s)")
        results = []
        for x in X:
            neighbors = self._get_neighbors(_x=x)
            prediction_counts = Counter(neighbor_label for neighbor_label, _ in neighbors)

            prediction = prediction_counts.most_common(1)[0][0]
            results.append(prediction)
        return results[0] if single_input else results

    def predict_probas(
        self,
        X: Union[
            Union[JointSeries, AngleSeries, FourierSeries],
            List[Union[JointSeries, AngleSeries, FourierSeries]],
        ],
    ) -> Union[List[Dict[str, float]], Dict[str, float]]:
        """Predict class probabilities for input data points.

        Args:
            X (Union[Union[JointSeries, AngleSeries, FourierSeries], List]):
                Input data point or list of input data points.

        Returns:
            List[Dict[str, float]]: List of dictionaries containing class probabilities.
        """
        if not isinstance(X, list):
            single_input = True
            X = [X]
        else:
            single_input = False
        logging.info(f"Predicting class label probabilities for input data: {len(X)} sample(s)")
        probas = []

        for x in X:
            neighbors = self._get_neighbors(x)
            total_neighbors = len(
                neighbors
            )  # Counting the number of neighbors for uniform weighting
            prediction_probs = {
                label: 0.0 for label in self.all_labels
            }  # Initialize with zero probabilities

            for neighbor_label, _ in neighbors:
                prediction_probs[neighbor_label] += 1.0  # Increment by 1 for each neighbor

            prediction_probs = {
                label: round(count / total_neighbors, 4)
                for label, count in prediction_probs.items()
            }

            probas.append(prediction_probs)

        return probas[0] if single_input else probas
