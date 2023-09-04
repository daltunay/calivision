import logging
from typing import List, Literal, Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..features import AngleSeries, JointSeries
from .model import ClassifierModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class LSTMModel(nn.Module):
    """LSTM-based Classifier.

    This class defines the architecture of an LSTM-based classifier for
    sequence classification tasks. It encapsulates the core model
    architecture and its forward pass.
    """

    def __init__(
        self,
        feature_type: Literal["joints", "angles", "fourier"],
        num_classes: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        """Initialize the LSTMModel object.

        Args:
            num_classes (int): Number of classes for classification.
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state of the LSTM.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.feature_type: Literal["joints", "angles", "fourier"] = feature_type
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.label_encoder = LabelEncoder()

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Get the output of the last time step
        return output


class LSTMClassifier(ClassifierModel):
    """Wrapper for Training and Using an LSTM Classifier.

    This class provides a higher-level interface for training an
    LSTMModel on input data and making predictions using the trained
    model. It separates concerns related to training and prediction from
    the core architecture of the LSTM model.
    """

    def __init__(
        self,
        feature_type: Literal["joints", "angles", "fourier"],
        num_classes: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        """Initialize the LSTMClassifier object.

        Args:
            num_classes (int): Number of classes for classification.
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state of the LSTM.
            num_layers (int): Number of LSTM layers.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimizer.
        """
        logging.info(
            f"Initializing LSTM classifier ({hidden_size=}, {num_layers=}, {num_epochs=}, {batch_size=}, {learning_rate=})"
        )
        self.feature_type = feature_type
        super().__init__(model_type="lstm", feature_type=self.feature_type)
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.label_encoder = LabelEncoder()

        self.criterion = nn.CrossEntropyLoss()
        self.model = LSTMModel(feature_type, num_classes, input_size, hidden_size, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _convert_to_tensors(self, X: List[Union[JointSeries, AngleSeries]]) -> torch.Tensor:
        """Convert a list of JointSeries or AngleSeries objects to a stacked tensor.

        Args:
            X (List[Union[JointSeries, AngleSeries]]): List of input data.

        Returns:
            torch.Tensor: Stacked tensor containing input data.
        """
        max_length = max(x.shape[0] for x in X)  # padding time series length
        tensors = []
        for x in X:
            x = x.reindex(range(max_length), fill_value=0)
            tensor = torch.Tensor(x.values)
            tensors.append(tensor)
        return torch.stack(tensors)

    def fit(self, X: List[Union[JointSeries, AngleSeries]], y: List[str]) -> None:
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        logging.info(f"Fitting LSTM classifier to training data: {len(X)} sample(s), {len(set(y))} unique labels")
        y_encoded = self.label_encoder.fit_transform(y)
        y_encoded_tensors = torch.LongTensor(y_encoded)

        X_tensors_stacked = self._convert_to_tensors(X)
        dataset = TensorDataset(X_tensors_stacked, y_encoded_tensors)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize epoch progress bar
        for epoch in tqdm(range(1, self.num_epochs + 1), desc="[IN PROGRESS] LSTM training", leave=True):
            epoch_loss = 0.0

            # Initialize batch progress bar
            batch_pbar = tqdm(train_loader, desc=f"{epoch=} | progression", leave=False)

            for num_batches, (X_batch, y_batch) in enumerate(batch_pbar, start=1):
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                avg_epoch_loss = epoch_loss / num_batches

                batch_pbar.set_postfix({"batch_loss": batch_loss, "avg_epoch_loss": avg_epoch_loss})

            batch_pbar.close()

        logging.info("[DONE] LSTM training")
        return self

    def predict(
        self, X: Union[List[Union[JointSeries, AngleSeries]], Union[JointSeries, AngleSeries]]
    ) -> Union[List[str], str]:
        """Predict class labels for input data.

        Args:
            X (Union[List[Union[JointSeries, AngleSeries]], Union[JointSeries, AngleSeries]]): Input data.

        Returns:
            Union[List[str], str]: Predicted class labels.
        """
        if not isinstance(X, list):
            single_input = True
            X = [X]
        else:
            single_input = False
        logging.info(f"Predicting class labels for input data: {len(X)} sample(s)")
        data_tensors = self._convert_to_tensors(X)
        predictions = []
        with torch.no_grad():
            for x in data_tensors:
                output = self.model(x.unsqueeze(0))
                _, predicted = torch.max(output, 1)
                predicted_labels = self.label_encoder.inverse_transform(predicted.numpy())
                predictions.append(predicted_labels[0])
        return predictions[0] if single_input else predictions

    def predict_probas(self, X: List[Union[JointSeries, AngleSeries]]) -> List[Dict[str, float]]:
        """Predict class probabilities for input data.

        Args:
            X (List[Union[JointSeries, AngleSeries]]): Input data.

        Returns:
            List[Dict[str, float]]: List of dictionaries containing class probabilities.
        """
        if not isinstance(X, list):
            single_input = True
            X = [X]
        else:
            single_input = False
        logging.info(f"Predicting class label probabilities for input data: {len(X)} sample(s)")
        data_tensors = self._convert_to_tensors(X)
        probas = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for x in data_tensors:
                output = self.model(x.unsqueeze(0))
                predicted_probs = softmax(output)
                class_probabilities = {
                    label: prob.item()
                    for label, prob in zip(self.label_encoder.classes_, predicted_probs[0])
                }
                probas.append(class_probabilities)
        return probas[0] if single_input else probas
