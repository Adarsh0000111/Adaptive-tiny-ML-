from sklearn.ensemble import RandomForestClassifier
from src.models.base import BaseIDSModel

class MediumIDSModel(BaseIDSModel):
    def __init__(self, n_estimators=25, max_depth=10, random_state=42, n_jobs=-1):
        super().__init__(
            name="Medium RandomForest",
            model_level="medium",
            est_latency_ms=1.45,
            est_ram_kb=48.0,
            est_energy_uJ=62.0
        )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
