from src.models.base import BaseIDSModel
from src.models.cnn import PacketCNNClassifier


class HeavyIDSModel(BaseIDSModel):
    """High laptop tier: 1D CNN over the ordered packet-feature vector."""

    def __init__(self, epochs=60, batch_size=256, learning_rate=0.0005,
                 patience=8, random_state=42):
        super().__init__(
            name="High 1D CNN",
            model_level="heavy",
            est_latency_ms=8.70,
            est_ram_kb=320.0,
            est_energy_uJ=340.0,
        )
        self.model = PacketCNNClassifier(epochs=epochs, batch_size=batch_size,
                                         learning_rate=learning_rate, patience=patience,
                                         random_state=random_state)

    def fit(self, X, y, validation_data=None):
        self.model.fit(X, y, validation_data=validation_data)
        return self
