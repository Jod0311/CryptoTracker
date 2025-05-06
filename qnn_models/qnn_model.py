"""Quantum Neural Network (QNN) model for cryptocurrency price prediction using PennyLane."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml

# Define the quantum device
dev = qml.device('lightning.qubit', wires=4)


@qml.qnode(dev)
def qnn(weights, x_input):
    """Quantum circuit representing the QNN."""
    qml.RX(x_input[0] * np.pi, wires=0)
    qml.RY(x_input[1] * np.pi, wires=1)
    qml.RX(weights[0], wires=2)
    qml.RY(weights[1], wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    return qml.expval(qml.PauliZ(0))


class QNNModel:
    """Class representing a simple Quantum Neural Network (QNN)."""

    def __init__(self, num_qubits=4, num_layers=2, learning_rate=0.01):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weights = np.random.randn(self.num_layers, self.num_qubits)
        self.optimizer = qml.GradientDescentOptimizer(stepsize=self.learning_rate)

    def train(self, x_data, labels, num_epochs=100):
        """Train the QNN model using gradient descent."""
        for epoch in range(num_epochs):
            for x, target in zip(x_data, labels):
                self.weights = self.optimizer.step(
                    lambda w: self._loss_function(w, x, target), self.weights
                )

            if epoch % 10 == 0:
                losses = [
                    self._loss_function(self.weights, x_i, y_i)
                    for x_i, y_i in zip(x_data, labels)
                ]
                avg_loss = np.mean(losses)
                print(f"Epoch {epoch}, Loss: {avg_loss}")

    def _loss_function(self, weights, x_input, target):
        """Mean squared error loss for QNN predictions."""
        prediction = qnn(weights.flatten(), x_input)
        return (prediction - target) ** 2

    def predict(self, x_data):
        """Predict values using the trained QNN."""
        return np.array([qnn(self.weights.flatten(), x) for x in x_data])


def train_qnn_model(df, coin_symbol):
    """Train a QNN for a specific cryptocurrency and predict the next price."""
    try:
        coin_df = df[df['symbol'] == coin_symbol].copy()
        coin_df.sort_values('last_updated', inplace=True)

        if len(coin_df) < 10:
            return None, "Not enough data to train QNN"

        prices = coin_df['current_price'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices)

        x_data, labels = [], []
        time_steps = 3
        for i in range(time_steps, len(prices_scaled)):
            x_data.append(prices_scaled[i - time_steps:i].flatten())
            labels.append(prices_scaled[i])

        x_data = np.array(x_data)
        labels = np.array(labels)

        qnn_model = QNNModel()
        qnn_model.train(x_data, labels, num_epochs=100)

        last_sequence = prices_scaled[-time_steps:].reshape(1, time_steps).flatten()
        predicted_scaled = qnn_model.predict([last_sequence])[0]
        predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

        return float(predicted_price), None

    except Exception as error:
        return None, f"Error in QNN: {str(error)}"
