from sklearn.tree import DecisionTreeClassifier
from src.models.base import BaseIDSModel

class LightweightIDSModel(BaseIDSModel):
    def __init__(self, max_depth=4, min_samples_leaf=5, criterion="gini", random_state=42):
        super().__init__(
            name="Lightweight DecisionTree",
            model_level="lightweight",
            est_latency_ms=0.12,
            est_ram_kb=4.5,
            est_energy_uJ=8.5
        )
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
