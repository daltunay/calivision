from typing import Literal


class ClassifierModel:
    def __init__(
        self,
        model_type: Literal["knn", "lstm"],
        feature_type: Literal["joints", "angles", "fourier"],
    ):
        self.model_type: Literal["knn", "lstm"] = model_type
        self.feature_type: Literal["joints", "angles", "fourier"] = feature_type

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method.")

    def predict_probas(self, X):
        raise NotImplementedError("Subclasses must implement this method.")
