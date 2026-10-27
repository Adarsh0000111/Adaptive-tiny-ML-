from src.config import load_config
from src.models.lightweight import LightweightIDSModel
from src.models.medium import MediumIDSModel
from src.models.heavy import HeavyIDSModel

def get_model(level, config_path="config.yaml"):
    config = load_config(config_path)
    m_cfg = config["models"].get(level, {})
    params = m_cfg.get("params", {})

    if level == "lightweight":
        return LightweightIDSModel(**params)
    elif level == "medium":
        return MediumIDSModel(**params)
    elif level == "heavy":
        return HeavyIDSModel(**params)
    else:
        raise ValueError(f"Unknown model level: {level}")
