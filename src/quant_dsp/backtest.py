import numpy as np
import pandas as pd

from .model import build_model, long_short_return
from .utils import compute_daily_ic


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Target5",
    train_days: int = 756,
    test_days: int = 126,
) -> tuple[pd.DataFrame, dict]:
    dates = sorted(df["Date"].unique())
    preds = []

    start = train_days
    while start + test_days < len(dates):
        train_start = dates[start - train_days]
        train_end = dates[start - 1]
        test_start = dates[start]
        test_end = dates[start + test_days - 1]

        tr = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)]
        te = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]

        if len(tr) < 1000 or len(te) < 100:
            start += test_days
            continue

        m = build_model()
        m.fit(tr[feature_cols], tr[target_col])
        p = m.predict(te[feature_cols])

        fold = te[["Date", "Ticker", target_col]].copy()
        fold["Pred"] = p
        preds.append(fold)

        start += test_days

    if not preds:
        raise ValueError("No backtest folds produced. Increase dataset length or lower train_days.")

    pred_df = pd.concat(preds, ignore_index=True)
    daily_ic = compute_daily_ic(pred_df, "Pred", target_col)

    ls = pred_df.groupby("Date", group_keys=False).apply(
        lambda x: long_short_return(x, "Pred", target_col)
    )

    metrics = {
        "mean_daily_rank_ic": float(np.nanmean(daily_ic)),
        "std_daily_rank_ic": float(np.nanstd(daily_ic)),
        "ic_ir": float(np.nanmean(daily_ic) / (np.nanstd(daily_ic) + 1e-9)),
        "mean_5d_ls_return": float(np.nanmean(ls)),
        "std_5d_ls_return": float(np.nanstd(ls)),
        "ls_sharpe_proxy": float(np.nanmean(ls) / (np.nanstd(ls) + 1e-9)),
        "n_prediction_rows": int(len(pred_df)),
        "n_prediction_days": int(pred_df["Date"].nunique()),
    }

    return pred_df, metrics
