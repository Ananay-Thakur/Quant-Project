import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_panel(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df


def save_metrics(metrics: dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))


def rank_ic(pred: pd.Series, target: pd.Series) -> float:
    mask = pred.notna() & target.notna()
    if mask.sum() < 5:
        return np.nan
    p = pred[mask].rank()
    t = target[mask].rank()
    return p.corr(t)


def compute_daily_ic(df: pd.DataFrame, pred_col: str, target_col: str) -> pd.Series:
    return df.groupby("Date", group_keys=False).apply(
        lambda x: rank_ic(x[pred_col], x[target_col])
    )
