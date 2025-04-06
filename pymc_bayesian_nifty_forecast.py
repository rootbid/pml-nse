import logging
import os
import pickle
import traceback
from datetime import timedelta

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

logging.basicConfig(level=logging.INFO)

def calculate_rsi(df, window=14):
    """Calculate the RSI for a given DataFrame."""
    
    # ✅ Price changes
    delta = df['close'].diff(1)

    # ✅ Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # ✅ Rolling averages for gain and loss
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    # ✅ Handle division by zero
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    
    # ✅ RSI calculation
    df['rsi'] = 100 - (100 / (1 + rs))

    # ✅ Remove NaNs at the beginning
    df['rsi'].fillna(50, inplace=True)  # RSI defaults to 50 in case of NaNs

    return df

def load_nifty_data(csv_file):
    """Load and preprocess NIFTY 50 data."""
    df = pd.read_csv(csv_file, parse_dates=['date'], index_col='date')
    df = df.sort_index()
    
    # Feature Engineering
    df['return'] = df['close'].pct_change()  # Daily return
    df['volatility'] = df['return'].rolling(window=14).std()  # Rolling volatility
    df['momentum'] = df['close'].diff(5)  # Momentum over 5 days
    
    # Moving averages
    df['sma'] = df['close'].rolling(window=20).mean()
    df['ema'] = df['close'].ewm(span=20).mean()
    
    df['sma'].fillna(df['sma'].mean(), inplace=True)      # Mean SMA
    df['ema'].fillna(df['ema'].mean(), inplace=True)      # Mean EMA
    df['volatility'].fillna(0, inplace=True)  
    df['momentum'].fillna(0, inplace=True)

    # RSI calculation
    df = calculate_rsi(df)
    
    logging.info(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(df)
    
    return df

def save_trained_models(model_name, model_dict, base_path="models"):
    """Save model trace and scalers only."""
    os.makedirs(base_path, exist_ok=True)

    try:
        model_dir = os.path.join(base_path, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)

        # Save trace
        trace_path = os.path.join(model_dir, "trace.pkl")
        with open(trace_path, "wb") as f:
            pickle.dump(model_dict["trace"], f)

        # Save scalers
        scalers_path = os.path.join(model_dir, "scalers.pkl")
        with open(scalers_path, "wb") as f:
            pickle.dump(model_dict["scalers"], f)

        logging.info(f"Saved {model_name} model data to {model_dir}")

    except Exception as e:
        logging.error(f"Error saving {model_name} model: {str(e)}")
        traceback.print_exc()


def load_trained_model(model_name, base_path="models"):
    """Load model trace and scalers."""
    try:
        model_dir = os.path.join(base_path, model_name.replace(" ", "_"))
        model_data = {}

        # Load trace
        trace_path = os.path.join(model_dir, "trace.pkl")
        if os.path.exists(trace_path):
            with open(trace_path, "rb") as f:
                model_data["trace"] = pickle.load(f)
        else:
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

        # Load scalers
        scalers_path = os.path.join(model_dir, "scalers.pkl")
        if os.path.exists(scalers_path):
            with open(scalers_path, "rb") as f:
                model_data["scalers"] = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scalers file not found: {scalers_path}")

        logging.info(f"Loaded {model_name} model from {model_dir}")
        return model_data

    except Exception as e:
        logging.error(f"Error loading {model_name} model: {str(e)}")
        traceback.print_exc()
        return None

def prepare_forecast_features(df, future_df, model_name, forecast_days):
    """Prepare features for forecasting based on model type."""
    # Get scaler values
    scalers = {}
    for col in ['close', 'volatility', 'momentum', 'sma', 'ema', 'rsi']:
        mean, std = df[col].mean(), df[col].std()
        scalers[col] = {'mean': mean, 'std': std}

    if model_name == 'Bayesian Linear':
        # Normalize features for Bayesian Linear model
        features = np.column_stack([
            (np.repeat(df['volatility'].iloc[-1], forecast_days) - scalers['volatility']['mean']) / scalers['volatility']['std'],
            (np.repeat(df['momentum'].iloc[-1], forecast_days) - scalers['momentum']['mean']) / scalers['momentum']['std'],
            (np.repeat(df['sma'].iloc[-1], forecast_days) - scalers['sma']['mean']) / scalers['sma']['std'],
            (np.repeat(df['ema'].iloc[-1], forecast_days) - scalers['ema']['mean']) / scalers['ema']['std'],
            (np.repeat(df['rsi'].iloc[-1], forecast_days) - scalers['rsi']['mean']) / scalers['rsi']['std'],
            np.arange(len(df), len(df) + forecast_days)/len(df)  # Normalized trend
        ])
        
    elif model_name == 'AR(1)':
        # For AR(1), we need the last observed value
        features = np.array([(df['close'].iloc[-1] - scalers['close']['mean']) / scalers['close']['std']])
        
    elif model_name == 'Hierarchical':
        # Normalize features for Hierarchical model
        features = np.column_stack([
            (np.repeat(df['volatility'].iloc[-1], forecast_days) - scalers['volatility']['mean']) / scalers['volatility']['std'],
            (np.repeat(df['momentum'].iloc[-1], forecast_days) - scalers['momentum']['mean']) / scalers['momentum']['std']
        ])
        # Add year index for hierarchical structure
        features = np.column_stack([
            features,
            np.repeat(pd.Categorical(df.index[-1].year).codes, forecast_days)
        ])
        
    elif model_name == 'GPR':
        # Normalize features for GPR model
        features = np.column_stack([
            (np.repeat(df['volatility'].iloc[-1], forecast_days) - scalers['volatility']['mean']) / scalers['volatility']['std'],
            (np.repeat(df['momentum'].iloc[-1], forecast_days) - scalers['momentum']['mean']) / scalers['momentum']['std']
        ])
        
    elif model_name == 'Dynamic Regression':
        # Normalize features for Dynamic Regression model
        features = np.column_stack([
            (np.repeat(df['volatility'].iloc[-1], forecast_days) - scalers['volatility']['mean']) / scalers['volatility']['std'],
            (np.repeat(df['ema'].iloc[-1], forecast_days) - scalers['ema']['mean']) / scalers['ema']['std'],
            np.arange(len(df), len(df) + forecast_days)/len(df)  # Normalized trend
        ])
    
    else:
        features = None
        logging.warning(f"Unknown model type: {model_name}")
    
    logging.info(f"Prepared forecast features for {model_name} model")
    return features

def plot_forecast(df, forecast_dates, pred_mean, pred_lower, pred_upper, model_name):
    """Plot forecasting results."""
    plt.figure(figsize=(14, 7))
    plt.title(f"{model_name} Model Forecast")
    plt.plot(df.index[-90:], df['close'][-90:], label='Historical', color='blue')
    plt.plot(forecast_dates, pred_mean, label='Forecast', color='red')
    plt.fill_between(forecast_dates, pred_lower, pred_upper,
                    color='red', alpha=0.2, label='95% CI')
    
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name.replace(' ', '_')}_forecast.png")
    plt.close()
    
    logging.info(f"Generated forecast plot for {model_name}")

def forecast_dynamic_regression(df, model_dict, forecast_days=10):
    """Generate forecasts for the Dynamic Regression model."""
    forecast_dates = [
        df.index[-1] + timedelta(days=i + 1) for i in range(forecast_days)
    ]

    # Ensure all required scalers are present
    required_features = ["volatility", "ema", "close"]
    scalers = model_dict["scalers"]

    # Create scalers if missing
    for feature in required_features:
        if feature not in scalers:
            mean = df[feature].mean()
            std = df[feature].std()
            scalers[feature] = {"mean": mean, "std": std}
            logging.info(f"Created missing scaler for {feature}")

    # Get last observed values and scale them
    scaler_vol = scalers["volatility"]
    scaler_ema = scalers["ema"]
    X_vol = (df["volatility"].iloc[-1] - scaler_vol["mean"]) / scaler_vol["std"]
    X_ema = (df["ema"].iloc[-1] - scaler_ema["mean"]) / scaler_ema["std"]
    
    # Get mean parameter values from trace
    trace = model_dict['trace']
    alpha_mean = float(trace.posterior['alpha'].mean())
    beta_vol_mean = float(trace.posterior['beta_vol'].mean())
    beta_ema_mean = float(trace.posterior['beta_ema'].mean())
    
    # Get the entire beta_t series
    beta_t_values = trace.posterior['beta_t'].mean(["chain", "draw"]).values
    
    # Get the last beta_t value from the training period
    beta_t_last = beta_t_values[-1]
    
    # Get the noise
    sigma_mean = float(trace.posterior['sigma'].mean())
    
    # Generate predictions
    predictions = []
    n_samples = 1000
    
    for _ in range(n_samples):
        forecast = []
        
        # Use the last beta_t value as a starting point and add random noise
        current_beta_t = beta_t_last
        
        for i in range(forecast_days):
            # Time trend for this step
            t = (len(df) + i) / len(df)
            
            # Prediction with noise
            mean = (alpha_mean + 
                   current_beta_t * t +
                   beta_vol_mean * X_vol + 
                   beta_ema_mean * X_ema)
            
            # Add noise to the prediction
            pred = np.random.normal(mean, sigma_mean)
            
            # Add noise to the beta_t value for the next step
            current_beta_t += np.random.normal(0, 0.01)  # Small noise
            
            forecast.append(pred)
        predictions.append(forecast)
    
    # Convert to array and denormalize
    predictions = np.array(predictions)
    scaler = model_dict['scalers']['close']
    predictions = predictions * scaler['std'] + scaler['mean']
    
    # Calculate statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)
    
    return forecast_dates, pred_mean, pred_lower, pred_upper

def forecast_bayesian_linear(df, model_dict, forecast_days=10):
    """Generate forecasts for the Bayesian Linear Regression model."""
    forecast_dates = [df.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Get last observed values and scale them
    scaler_vol = model_dict['scalers']['volatility']
    scaler_mom = model_dict['scalers']['momentum']
    scaler_sma = model_dict['scalers']['sma']
    scaler_ema = model_dict['scalers']['ema']
    scaler_rsi = model_dict['scalers']['rsi']
    
    X_vol = (df['volatility'].iloc[-1] - scaler_vol['mean']) / scaler_vol['std']
    X_mom = (df['momentum'].iloc[-1] - scaler_mom['mean']) / scaler_mom['std']
    X_sma = (df['sma'].iloc[-1] - scaler_sma['mean']) / scaler_sma['std']
    X_ema = (df['ema'].iloc[-1] - scaler_ema['mean']) / scaler_ema['std']
    X_rsi = (df['rsi'].iloc[-1] - scaler_rsi['mean']) / scaler_rsi['std']
    
    # Get mean parameter values from trace
    trace = model_dict['trace']
    alpha_mean = float(trace.posterior['alpha'].mean())
    beta_vol_mean = float(trace.posterior['beta_vol'].mean())
    beta_mom_mean = float(trace.posterior['beta_mom'].mean())
    beta_sma_mean = float(trace.posterior['beta_sma'].mean())
    beta_ema_mean = float(trace.posterior['beta_ema'].mean())
    beta_rsi_mean = float(trace.posterior['beta_rsi'].mean())
    beta_trend_mean = float(trace.posterior['beta_trend'].mean())
    sigma_mean = float(trace.posterior['sigma'].mean())
    
    # Generate predictions
    predictions = []
    n_samples = 1000
    
    for _ in range(n_samples):
        forecast = []
        for i in range(forecast_days):
            # Time trend for this step
            t = (len(df) + i) / len(df)
            
            # Prediction with noise
            mean = (alpha_mean + 
                   beta_vol_mean * X_vol +
                   beta_mom_mean * X_mom +
                   beta_sma_mean * X_sma +
                   beta_ema_mean * X_ema +
                   beta_rsi_mean * X_rsi +
                   beta_trend_mean * t)
            
            pred = np.random.normal(mean, sigma_mean)
            forecast.append(pred)
        predictions.append(forecast)
    
    # Convert to array and denormalize
    predictions = np.array(predictions)
    scaler = model_dict['scalers']['close']
    predictions = predictions * scaler['std'] + scaler['mean']
    
    # Calculate statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)
    
    return forecast_dates, pred_mean, pred_lower, pred_upper

def forecast_ar1(df, model_dict, forecast_days=10):
    """Generate forecasts for the AR(1) model."""
    forecast_dates = [df.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Get last observed value and scale it
    scaler = model_dict['scalers']['close']
    last_close_scaled = (df['close'].iloc[-1] - scaler['mean']) / scaler['std']
    
    # Get mean parameter values from trace
    trace = model_dict['trace']
    alpha_mean = float(trace.posterior['alpha'].mean())
    beta_mean = float(trace.posterior['beta'].mean())
    sigma_mean = float(trace.posterior['sigma'].mean())
    
    # Generate predictions
    predictions = []
    n_samples = 1000
    
    for _ in range(n_samples):
        forecast = []
        
        # Use the last observed value as a starting point
        current_value = last_close_scaled
        
        for i in range(forecast_days):
            # Prediction with noise
            mean = alpha_mean + beta_mean * current_value
            pred = np.random.normal(mean, sigma_mean)
            
            # Update the current value for the next step
            current_value = pred
            
            forecast.append(pred)
        predictions.append(forecast)
    
    # Convert to array and denormalize
    predictions = np.array(predictions)
    predictions = predictions * scaler['std'] + scaler['mean']
    
    # Calculate statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)
    
    return forecast_dates, pred_mean, pred_lower, pred_upper

def forecast_hierarchical(df, model_dict, forecast_days=10):
    """Generate forecasts for the Hierarchical model."""
    forecast_dates = [df.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Get last observed values and scale them
    scaler_vol = model_dict['scalers']['volatility']
    scaler_mom = model_dict['scalers']['momentum']
    
    X_vol = (df['volatility'].iloc[-1] - scaler_vol['mean']) / scaler_vol['std']
    X_mom = (df['momentum'].iloc[-1] - scaler_mom['mean']) / scaler_mom['std']
    
    # Get mean parameter values from trace
    trace = model_dict['trace']
    alpha_mean = float(trace.posterior['alpha'].mean())
    beta_vol_mean = float(trace.posterior['beta_vol'].mean())
    beta_mom_mean = float(trace.posterior['beta_mom'].mean())
    sigma_mean = float(trace.posterior['sigma'].mean())
    
    # Get year effect for the current year
    current_year_idx = pd.Categorical([df.index[-1].year]).codes[0]
    year_effects = trace.posterior['year_effect'].mean(["chain", "draw"]).values
    year_effect_mean = year_effects[current_year_idx]
    
    # Generate predictions
    predictions = []
    n_samples = 1000
    
    for _ in range(n_samples):
        forecast = []
        for i in range(forecast_days):
            # Prediction with noise
            mean = (alpha_mean + 
                   year_effect_mean +
                   beta_vol_mean * X_vol + 
                   beta_mom_mean * X_mom)
            
            pred = np.random.normal(mean, sigma_mean)
            forecast.append(pred)
        predictions.append(forecast)
    
    # Convert to array and denormalize
    predictions = np.array(predictions)
    scaler = model_dict['scalers']['close']
    predictions = predictions * scaler['std'] + scaler['mean']
    
    # Calculate statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)
    
    return forecast_dates, pred_mean, pred_lower, pred_upper

def forecast_gpr(df, model_dict, forecast_days=10):
    """Generate forecasts using optimized Gaussian Process Regression."""
    forecast_dates = [
        df.index[-1] + timedelta(days=i + 1) for i in range(forecast_days)
    ]

    # Get scalers and normalize features
    scaler_vol = model_dict["scalers"]["volatility"]
    scaler_mom = model_dict["scalers"]["momentum"]
    scaler_close = model_dict["scalers"]["close"]

    X_train = np.column_stack(
        [
            (df["volatility"].values - scaler_vol["mean"]) / scaler_vol["std"],
            (df["momentum"].values - scaler_mom["mean"]) / scaler_mom["std"],
        ]
    ).astype(np.float64)

    y_train = (
        (df["close"].values - scaler_close["mean"]) / scaler_close["std"]
    ).astype(np.float64)

    # Get model parameters
    trace = model_dict["trace"]
    ls_mean = float(trace.posterior["ls"].mean())
    eta_mean = float(trace.posterior["eta"].mean())
    sigma_mean = float(trace.posterior["sigma"].mean())

    # Kernel function
    def kernel(X1, X2, ls, eta):
        dist = ((X1[:, None, :] - X2[None, :, :]) / ls) ** 2
        return eta**2 * np.exp(-0.5 * np.sum(dist, axis=-1))

    # Precompute kernel matrices
    K_train = kernel(X_train, X_train, ls_mean, eta_mean)
    K_train += sigma_mean**2 * np.eye(len(X_train))
    K_train_inv = np.linalg.inv(K_train)

    # Initialize predictions
    n_samples = 500
    batch_size = 50
    predictions = np.zeros((n_samples, forecast_days))

    # Initial features
    X_current = np.array(
        [
            [
                (df["volatility"].iloc[-1] - scaler_vol["mean"]) / scaler_vol["std"],
                (df["momentum"].iloc[-1] - scaler_mom["mean"]) / scaler_mom["std"],
            ]
        ]
    )

    for batch in range(0, n_samples, batch_size):
        batch_end = min(batch + batch_size, n_samples)
        batch_size_current = batch_end - batch

        # Initialize batch predictions
        forecast_batch = np.zeros((batch_size_current, forecast_days))
        X_batch = np.tile(X_current, (batch_size_current, 1))  # Replicate for batch

        for i in range(forecast_days):
            # Compute kernel matrices
            K_test = kernel(X_batch, X_train, ls_mean, eta_mean)
            K_test_test = kernel(X_batch, X_batch, ls_mean, eta_mean)

            # Compute posterior mean and variance
            post_mean = K_test @ K_train_inv @ y_train
            post_var = np.maximum(
                np.diag(K_test_test - K_test @ K_train_inv @ K_test.T), 1e-8
            )

            # Generate samples
            forecast_batch[:, i] = np.random.normal(post_mean, np.sqrt(post_var))

            # Update features
            X_batch = np.column_stack(
                [
                    X_batch[:, 0] * 0.95 + np.random.normal(0, 0.1, batch_size_current),
                    X_batch[:, 1] * 0.95 + np.random.normal(0, 0.1, batch_size_current),
                ]
            )

        predictions[batch:batch_end] = forecast_batch

    # Denormalize predictions
    predictions = predictions * scaler_close["std"] + scaler_close["mean"]

    # Calculate statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_lower = np.percentile(predictions, 2.5, axis=0)
    pred_upper = np.percentile(predictions, 97.5, axis=0)

    return forecast_dates, pred_mean, pred_lower, pred_upper

def train_and_evaluate_models(df, forecast_days=10):
    """Train and compare multiple Bayesian models."""
    
    results = {}

    # Normalize features for better convergence
    df_norm = df.copy()
    scalers = {}
    for col in ['close', 'volatility', 'momentum', 'sma', 'ema', 'rsi']:
        mean, std = df[col].mean(), df[col].std()
        df_norm[col] = (df[col] - mean) / std
        scalers[col] = {'mean': mean, 'std': std}

    # 1. Bayesian Linear Regression
    logging.info("Training Bayesian Linear Regression...")
    with pm.Model() as model1:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_vol = pm.Normal('beta_vol', mu=0, sigma=2)
        beta_mom = pm.Normal('beta_mom', mu=0, sigma=2)
        beta_sma = pm.Normal('beta_sma', mu=0, sigma=2)
        beta_ema = pm.Normal('beta_ema', mu=0, sigma=2)
        beta_rsi = pm.Normal('beta_rsi', mu=0, sigma=2)
        beta_trend = pm.Normal('beta_trend', mu=0, sigma=2)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Features
        X_vol = df_norm['volatility']
        X_mom = df_norm['momentum']
        X_sma = df_norm['sma']
        X_ema = df_norm['ema']
        X_rsi = df_norm['rsi']
        X_trend = np.arange(len(df_norm))/len(df_norm)
        
        # Linear combination
        mu = (alpha + 
              beta_vol * X_vol +
              beta_mom * X_mom +
              beta_sma * X_sma +
              beta_ema * X_ema +
              beta_rsi * X_rsi +
              beta_trend * X_trend)
        
        # Likelihood
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=df_norm['close'])
        
        # Inference
        trace1 = pm.sample(1000, tune=500, chains=4, target_accept=0.9)
        results['Bayesian Linear'] = {'model': model1, 'trace': trace1, 'scalers': scalers}
        logging.info("Bayesian Linear Regression model trained.")
        # Plot trace
        pm.plot_trace(trace1)
        plt.savefig("traceplot_bayesian_linear.png")
        plt.close()
        # Plot posterior predictive checks
        pm.plot_posterior(trace1)
        plt.savefig("posterior_predictive_bayesian_linear.png")
        plt.close()
        # Save the model
        save_trained_models(
            "Bayesian Linear", results["Bayesian Linear"], base_path="models"
        )

    # 2. AR(1) Model
    logging.info("Training AR(1) Model...")
    with pm.Model() as model2:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        ar_term = pm.AR('ar_term', beta, sigma=sigma, observed=df_norm['close'])
        
        trace2 = pm.sample(2000, tune=1000, chains=4)
        results['AR(1)'] = {'model': model2, 'trace': trace2, 'scalers': scalers}
        logging.info("AR(1) model trained.")
        # Plot trace
        pm.plot_trace(trace2)
        plt.savefig("traceplot_ar1.png")
        plt.close()
        
        # Plot posterior predictive checks
        pm.plot_posterior(trace2)
        plt.savefig("posterior_predictive_ar1.png")
        plt.close()

        # Save the model
        save_trained_models(
            "AR(1)", results["AR(1)"], base_path="models"
        )

    # 3. Hierarchical Model
    logging.info("Training Hierarchical Model...")
    with pm.Model() as model3:
        # Hierarchical priors
        sigma_year = pm.HalfNormal('sigma_year', sigma=1)
        year_idx = pd.Categorical(df.index.year).codes
        
        # Global parameters
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        year_effect = pm.Normal('year_effect', mu=0, sigma=sigma_year, shape=len(set(year_idx)))
        
        # Individual coefficients instead of vector
        beta_vol = pm.Normal('beta_vol', mu=0, sigma=2)
        beta_mom = pm.Normal('beta_mom', mu=0, sigma=2)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Features
        X_vol = df_norm['volatility']
        X_mom = df_norm['momentum']
        
        # Model specification with individual terms
        mu = (alpha + 
              year_effect[year_idx] + 
              beta_vol * X_vol + 
              beta_mom * X_mom)
        
        # Likelihood
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=df_norm['close'])
        
        # Inference
        trace3 = pm.sample(draws=2000,
            tune=1000,
            chains=4,
            return_inferencedata=True,
            random_seed=42)
        results['Hierarchical'] = {'model': model3, 'trace': trace3, 'scalers': scalers}
        logging.info("Hierarchical model trained.")
        # Plot trace
        pm.plot_trace(trace3)
        plt.savefig("traceplot_hierarchical.png")
        plt.close()
        # Plot posterior predictive checks
        pm.plot_posterior(trace3)
        plt.savefig("posterior_predictive_hierarchical.png")
        plt.close()

        # Save the model
        save_trained_models(
            "Hierarchical", results["Hierarchical"], base_path="models"
        )

    # 4. Gaussian Process Regression Model
    logging.info("Training Gaussian Process Regression Model...")
    with pm.Model() as model4:
        # Feature Engineering
        X_train = np.column_stack([
            df['volatility'].values,
            df['momentum'].values,
        ]).astype(np.float64)

        y_train = df['close'].values.astype(np.float64)
        
        # Normalize Features
        scalers = {
            'volatility': {'mean': X_train[:, 0].mean(), 'std': X_train[:, 0].std()},
            'momentum': {'mean': X_train[:, 1].mean(), 'std': X_train[:, 1].std()},
            'close': {'mean': y_train.mean(), 'std': y_train.std()}
        }
        
        X_train = np.column_stack([
            (X_train[:, 0] - scalers['volatility']['mean']) / scalers['volatility']['std'],
            (X_train[:, 1] - scalers['momentum']['mean']) / scalers['momentum']['std'],
        ])
        
        y_train = (y_train - scalers['close']['mean']) / scalers['close']['std']

        # Priors for kernel hyperparameters
        ls = pm.Gamma("ls", alpha=2, beta=1)  # Length Scale
        eta = pm.HalfCauchy("eta", beta=1)    # Signal Variance
        sigma = pm.HalfNormal("sigma", sigma=1.0)  # Observation noise
        
        # Matern 3/2 Kernel
        cov_func = pm.gp.cov.Matern32(input_dim=2, ls=ls)
        K = eta**2 * cov_func
        
        # Define GP with sparse approximation
        gp = pm.gp.MarginalSparse(cov_func=K)
        
        # Select inducing points (subset of training points)
        n_inducing = min(100, len(X_train))
        idx = np.linspace(0, len(X_train)-1, n_inducing, dtype=int)
        Xu = X_train[idx]
        
        # Likelihood
        _ = gp.marginal_likelihood(
            "y",
            X=X_train,
            Xu=Xu,
            y=y_train,
            noise=sigma,
            jitter=1e-6
        )
        
        # Sample
        trace4 = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True
        )
        
        results['GPR'] = {'model': model4, 'trace': trace4, 'scalers': scalers}
        logging.info("GPR model trained.")
        # Plot trace
        pm.plot_trace(trace4)
        plt.savefig("traceplot_gpr.png")
        plt.close()
        # Plot posterior predictive checks
        pm.plot_posterior(trace4)
        plt.savefig("posterior_predictive_gpr.png")
        plt.close()

        # Save the model
        save_trained_models(
            "GPR", results["GPR"], base_path="models"
        )

    # 5. Dynamic Regression Model
    logging.info("Training Dynamic Regression Model...")
    with pm.Model() as model5:
        # Dimensions as shared variables
        n_points = pm.Data('n_points', len(df_norm))
        # Time-varying coefficients
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        
        # Random walk priors for dynamic coefficients
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.1)
        
        # Base coefficients
        beta_vol = pm.Normal('beta_vol', mu=0, sigma=2)
        beta_ema = pm.Normal('beta_ema', mu=0, sigma=2)
        
        # Time-varying trend coefficient
        beta_t = pm.GaussianRandomWalk(
            'beta_t', 
            sigma=sigma_beta,
            shape=len(df_norm),
            init_dist=pm.Normal.dist(mu=0, sigma=1)
        )
        
        # Features as shared variables with fixed shapes
        X_vol = pm.Data('X_vol', df_norm['volatility'].values)
        X_ema = pm.Data('X_ema', df_norm['ema'].values)
        trend = pm.Data('trend', np.arange(len(df_norm))/len(df_norm))
        
        # Combined model using indexing for time-varying effects
        mu = (alpha + 
              beta_t * trend +  # Time-varying trend effect
              beta_vol * X_vol + 
              beta_ema * X_ema)
        
        # Observation noise
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=df_norm['close'])

        # Inference with improved settings
        trace5 = pm.sample(
            draws=4000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            return_inferencedata=True,
            random_seed=42,
            init='adapt_diag'
        )
        
        results['Dynamic Regression'] = {'model': model5, 'trace': trace5, 'scalers': scalers}
        logging.info("Dynamic Regression model trained.")
        # Plot trace
        pm.plot_trace(trace5)
        plt.savefig("traceplot_dynamic_regression.png")
        plt.close()
        # Plot posterior predictive checks
        pm.plot_posterior(trace5)
        plt.savefig("posterior_predictive_dynamic_regression.png")
        plt.close()
        # Save the model
        save_trained_models(
            "Dynamic Regression", results["Dynamic Regression"], base_path="models"
        )

    # # Load model from saved files
    # logging.info("Loading GPR model from saved files...")
    # gpr_model_data = load_trained_model('GPR')
    # if gpr_model_data is not None:
    #     results['GPR'] = gpr_model_data
    #     logging.info("GPR model loaded successfully")
    # else:
    #     logging.error("Failed to load GPR model")
    
    # Forecasting with all models
    logging.info("Generating forecasts for all models...")
    for model_name, model_dict in results.items():
        try:
            if model_name == 'Bayesian Linear':
                forecast_dates, pred_mean, pred_lower, pred_upper = forecast_bayesian_linear(
                    df, model_dict, forecast_days
                )
                plot_forecast(df, forecast_dates, pred_mean, pred_lower, pred_upper, model_name)
            elif model_name == 'Dynamic Regression':
                forecast_dates, pred_mean, pred_lower, pred_upper = forecast_dynamic_regression(
                    df, model_dict, forecast_days
                )
                plot_forecast(df, forecast_dates, pred_mean, pred_lower, pred_upper, model_name)
            elif model_name == 'AR(1)':
                forecast_dates, pred_mean, pred_lower, pred_upper = forecast_ar1(
                    df, model_dict, forecast_days
                )
                plot_forecast(df, forecast_dates, pred_mean, pred_lower, pred_upper, model_name)
            elif model_name == 'Hierarchical':
                forecast_dates, pred_mean, pred_lower, pred_upper = forecast_hierarchical(
                    df, model_dict, forecast_days
                )
                plot_forecast(df, forecast_dates, pred_mean, pred_lower, pred_upper, model_name)
            elif model_name == 'GPR':
                forecast_dates, pred_mean, pred_lower, pred_upper = forecast_gpr(
                    df, model_dict, forecast_days
                )
                plot_forecast(df, forecast_dates, pred_mean, pred_lower, pred_upper, model_name)            
        except Exception as e:
            logging.error(f"Error forecasting with {model_name} model: {str(e)}")
            traceback.print_exc()
            continue

    return results

if __name__ == "__main__":
    # Load and process data
    df = load_nifty_data("data/nifty_data.csv")
    
    # Train models
    results = train_and_evaluate_models(df)

