import yaml
from src.models.lightweight import LightweightIDSModel
from src.models.medium import MediumIDSModel
from src.models.heavy import HeavyIDSModel

def get_model(level, config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
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
