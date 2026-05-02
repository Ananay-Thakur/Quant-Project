import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert


def _wavelet_denoise(series: pd.Series, wavelet: str = "db4", level: int = 1) -> pd.Series:
    values = np.array(series.fillna(0.0).values, dtype=float, copy=True)
    coeffs = pywt.wavedec(values, wavelet, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 0.0
    threshold = sigma * np.sqrt(2 * np.log(len(values) + 1))
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    recon = pywt.waverec(coeffs, wavelet, mode="symmetric")
    recon = recon[: len(values)]
    return pd.Series(recon, index=series.index)


def _spectral_entropy(window_vals: np.ndarray) -> float:
    if len(window_vals) < 8:
        return np.nan
    x = window_vals - np.nanmean(window_vals)
    fft = np.fft.rfft(x)
    psd = np.abs(fft) ** 2
    total = psd.sum()
    if total <= 0:
        return np.nan
    p = psd / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(psd)))


def make_dsp_features(df: pd.DataFrame, ret_col: str = "Ret1") -> pd.DataFrame:
    out = []
    for ticker, g in df.groupby("Ticker"):
        g = g.sort_values("Date").copy()

        g["RetDenoised"] = _wavelet_denoise(g[ret_col])
        g["TrendLF"] = g["RetDenoised"].rolling(20).mean()
        g["TrendMF"] = g["RetDenoised"].rolling(10).mean() - g["RetDenoised"].rolling(40).mean()
        g["OscHF"] = g["RetDenoised"] - g["RetDenoised"].rolling(5).mean()

        amp = np.abs(hilbert(g[ret_col].fillna(0.0).values))
        g["HilbertAmp"] = amp

        g["SpecEntropy20"] = g[ret_col].rolling(20).apply(
            lambda x: _spectral_entropy(x.values), raw=False
        )
        g["VolZ20"] = (g["Volume"] - g["Volume"].rolling(20).mean()) / (g["Volume"].rolling(20).std() + 1e-9)

        out.append(g)

    return pd.concat(out, ignore_index=True)


def make_physics_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    out = []
    for ticker, g in df.groupby("Ticker"):
        g = g.sort_values("Date").copy()

        x = np.log(g[price_col].clip(lower=1e-8))
        v = x.diff()
        a = v.diff()

        v_lag = v.shift(1)
        eps = 1e-9
        gamma = -a / (v_lag.replace(0, np.nan) + eps)
        gamma = gamma.clip(-10, 10)

        sigma = v.rolling(20).std()
        force = a + gamma.fillna(0.0) * v

        potential = -x.rolling(30).mean()
        curvature = potential.diff().diff()

        kinetic = 0.5 * (v ** 2)
        total_energy_imb = kinetic + potential

        g["Velocity"] = v
        g["Acceleration"] = a
        g["FrictionGamma"] = gamma
        g["DiffusionSigma"] = sigma
        g["ForceProxy"] = force
        g["Potential"] = potential
        g["Curvature"] = curvature
        g["EnergyImbalance"] = total_energy_imb

        out.append(g)

    return pd.concat(out, ignore_index=True)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Ticker", "Date"])
    df["Ret1"] = df.groupby("Ticker")["Adj Close"].pct_change()
    df["Target5"] = df.groupby("Ticker")["Adj Close"].pct_change(5).shift(-5)

    dsp = make_dsp_features(df)
    phy = make_physics_features(dsp)

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

    clean = phy.dropna(subset=feature_cols + ["Target5"]).copy()
    return clean
