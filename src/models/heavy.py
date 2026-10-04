from sklearn.neural_network import MLPClassifier
from src.models.base import BaseIDSModel

class HeavyIDSModel(BaseIDSModel):
    def __init__(self, hidden_layer_sizes=(128, 64, 32), max_iter=100, activation="relu", solver="adam", random_state=42):
        super().__init__(
            name="Heavy Deep MLP",
            model_level="heavy",
            est_latency_ms=8.70,
            est_ram_kb=320.0,
            est_energy_uJ=340.0
        )
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            activation=activation,
            solver=solver,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
