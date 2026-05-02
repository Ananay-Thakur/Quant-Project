from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quant_dsp.backtest import walk_forward_backtest
from src.quant_dsp.features import build_feature_table
from src.quant_dsp.utils import save_metrics


def main() -> None:
    data_path = Path("data/us_largecap_10y_daily.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "Data file missing. Run: python scripts/download_data.py"
        )

    df = pd.read_csv(data_path, parse_dates=["Date"])
    features = build_feature_table(df)

    feature_cols = [
        "RetDenoised",
        "TrendLF",
        "TrendMF",
        "OscHF",
        "HilbertAmp",
        "SpecEntropy20",
        "VolZ20",
        "Velocity",
        "Acceleration",
        "FrictionGamma",
        "DiffusionSigma",
        "ForceProxy",
        "Potential",
        "Curvature",
        "EnergyImbalance",
    ]

    pred_df, metrics = walk_forward_backtest(features, feature_cols=feature_cols)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    save_metrics(metrics, str(out_dir / "metrics.json"))

    print("Backtest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
