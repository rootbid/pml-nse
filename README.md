# NIFTY 50 Forecasting using Bayesian Models

A comprehensive Python implementation for forecasting NIFTY 50 stock prices using multiple Bayesian modeling approaches. This project leverages PyMC for Bayesian inference and includes various model architectures for robust prediction.

## 🚀 Features

- Multiple Bayesian model implementations
- Technical indicator generation
- GPU acceleration support
- Model persistence
- Forecast visualization
- Uncertainty quantification

## 📊 Models Implemented

1. **Bayesian Linear Regression**
   - Multiple technical indicators
   - Trend component
   - Global parameter estimation

2. **AR(1) Model**
   - Simple autoregressive structure
   - Single lag dependency
   - Basic time series patterns

3. **Hierarchical Model**
   - Year-level grouping
   - Volatility and momentum features
   - Temporal clustering support

4. **Gaussian Process Regression (GPR)**
   - Nonlinear relationships
   - Sparse approximation
   - Matern 3/2 kernel

5. **Dynamic Regression**
   - Time-varying coefficients
   - Random walk evolution
   - Adaptive trend modeling

## 🛠️ Technical Indicators

- RSI (Relative Strength Index)
- Volatility (14-day rolling)
- Momentum (5-day difference)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)

## 📈 Features

### Data Processing
- Automatic feature engineering
- Data normalization
- Missing value handling
- Time series preprocessing

### Model Training
- MCMC sampling with PyMC
- GPU acceleration support
- Proper chain initialization
- Convergence monitoring

### Forecasting
- Multi-step ahead predictions
- Uncertainty quantification
- Confidence intervals
- Forecast visualization

## 🔧 Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pymc numpy pandas matplotlib jax
```

## 💻 Usage

```python
# Load and process data
df = load_nifty_data("data/nifty_data.csv")

# Train models
results = train_and_evaluate_models(df)

# Save trained models
save_trained_models(results)
```

## 📊 Model Persistence

Models are automatically saved with:
- Model parameters
- MCMC traces
- Scaling parameters

## 📉 Visualization

Each model generates:
- Historical vs predicted plots
- Confidence intervals
- Trend analysis
- Forecast evaluation

## 🔬 Technical Details

- **Framework**: PyMC for Bayesian modeling
- **Acceleration**: JAX for GPU support
- **Visualization**: Matplotlib
- **Data Processing**: Pandas, NumPy

## 📝 License

MIT License

## 🤝 Contributing

Feel free to open issues and pull requests for improvements and bug fixes.

## 📚 References

- [PyMC Documentation](https://docs.pymc.io/)
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [Time Series Analysis by State Space Methods](https://www.ssfpack.com/DKbook.html)