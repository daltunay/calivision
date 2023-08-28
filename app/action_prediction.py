import json
import logging

from flask import Response, render_template
import torch
import os
from typing import Dict, Union

from src.features import JointSeries, AngleSeries, FourierSeries
from src.models import kNNClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class ActionRecognitionApp:
    """Class to manage the action recognition."""

    def __init__(self, model_name):
        """Initialize the app instance with necessary attributes."""
        self.model: kNNClassifier = torch.load(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", model_name)
        )
        self.predictions: Dict[str, float] = {}

    def predict(self, X: Union[JointSeries, AngleSeries, FourierSeries]):
        self.predictions = self.model.predict_probas(X)

    def visualize_predictions(self):
        if self.predictions is None:
            return "Predictions not available."

        return render_template("visualize_predictions.html", predictions=self.predictions)
