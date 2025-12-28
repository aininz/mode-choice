from dataclasses import dataclass
from typing import List, Dict

MODES: List[str] = ["train", "car", "bus", "air"]
ALT2ID: Dict[str, int] = {m: i for i, m in enumerate(MODES)}
ID2ALT: Dict[int, str] = {i: m for m, i in ALT2ID.items()}
N_ITEMS: int = len(MODES)

@dataclass(frozen=True)
class ModelConfig:
    base_item_features: List[str] = None
    item_features: List[str] = None
    case_features: List[str] = None

    # Nested structure: air vs land
    air_modes: List[str] = None
    land_modes: List[str] = None

    def __post_init__(self):
        object.__setattr__(self, "base_item_features", self.base_item_features or ["cost", "ivt", "ovt", "freq"])
        object.__setattr__(self, "item_features", self.item_features or list(self.base_item_features))
        object.__setattr__(self, "case_features", self.case_features or ["income", "urban"])
        object.__setattr__(self, "air_modes", self.air_modes or ["air"])
        object.__setattr__(self, "land_modes", self.land_modes or ["train", "car", "bus"])