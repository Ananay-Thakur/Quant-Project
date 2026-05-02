# Quantitative DSP + Physics-Informed Alpha Research Starter

This project builds a research pipeline that combines:
- Digital signal processing features on market time series
- Physics-inspired state variables (velocity, friction, force proxy)
- Cross-sectional alpha modeling and walk-forward evaluation

## Quickstart

1. Create environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Download data (US large-cap basket):

```powershell
python scripts\download_data.py
```

3. Run research pipeline:

```powershell
python scripts\run_research.py
```

4. Launch interactive dashboard:

```powershell
python -m streamlit run dashboard\app.py
```

## Streamlit Community Cloud

- Repository must be on GitHub.
- Entrypoint file for deployment: `streamlit_app.py` (root).
- Branch: `main` (recommended).

Outputs are written to `outputs/`.

## Notes

- This is a research baseline, not production trading advice.
- Add transaction cost, borrow constraints, and robust risk controls before live usage.
