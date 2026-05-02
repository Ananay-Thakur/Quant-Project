from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class ModelConfig:
    n_estimators: int = 60
    max_depth: int = 5
    random_state: int = 42


def build_model(cfg: ModelConfig | None = None) -> Pipeline:
    cfg = cfg or ModelConfig()
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    random_state=cfg.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return model


def long_short_return(g: pd.DataFrame, pred_col: str, target_col: str, q: float = 0.2) -> float:
    n = len(g)
    if n < 20:
        return np.nan
    k = max(1, int(n * q))
    s = g.sort_values(pred_col)
    short = s.iloc[:k][target_col].mean()
    long = s.iloc[-k:][target_col].mean()
    return float(long - short)
