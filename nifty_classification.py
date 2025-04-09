import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load the NIFTY data from CSV file"""
    logger.info("Loading data...")
    try:
        df = pd.read_csv("data/nifty_data.csv", parse_dates=["date"], index_col="date")
        logger.info(f"Data loaded, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    logger.info("Adding technical indicators...")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Basic price info
    df["return"] = df["close"].pct_change() * 100
    df["volatility"] = df["return"].rolling(window=20).std()

    # Target 1: Next day direction (1 if next day close > today's close)
    df["next_day_direction"] = df["close"].shift(-1) > df["close"]
    df["next_day_direction"] = df["next_day_direction"].astype(int)

    # Target 2: Market regime (using 50-day SMA and volatility)
    # First calculate SMA and volatility
    df["sma_50"] = df["close"].rolling(window=50).mean()

    # Define conditions for market regimes:
    # 0 = Bear (price below SMA50 and high volatility)
    # 1 = Sideways (price near SMA50)
    # 2 = Bull (price above SMA50 and low volatility)

    # Define thresholds
    price_threshold = 0.02  # 2% from SMA
    vol_threshold = df["volatility"].quantile(0.7)

    # Calculate price distance from SMA as percentage
    df["sma_dist"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    # Initialize regime column
    df["market_regime"] = 1  # Default to sideways

    # Bearish: price significantly below SMA and higher volatility
    bear_mask = (df["sma_dist"] < -price_threshold) & (df["volatility"] > vol_threshold)
    df.loc[bear_mask, "market_regime"] = 0

    # Bullish: price significantly above SMA
    bull_mask = df["sma_dist"] > price_threshold
    df.loc[bull_mask, "market_regime"] = 2

    # Feature 1: Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()

    # SMA crossovers and ratios
    df["sma_5_20_cross"] = df["sma_5"] > df["sma_20"]
    df["sma_20_50_cross"] = df["sma_20"] > df["sma_50"]
    df["sma_5_20_ratio"] = df["sma_5"] / df["sma_20"]
    df["sma_20_50_ratio"] = df["sma_20"] / df["sma_50"]

    # Feature 2: Bollinger Bands
    for window in [20]:
        df[f"bb_middle_{window}"] = df["close"].rolling(window=window).mean()
        df[f"bb_std_{window}"] = df["close"].rolling(window=window).std()
        df[f"bb_upper_{window}"] = df[f"bb_middle_{window}"] + (
            2 * df[f"bb_std_{window}"]
        )
        df[f"bb_lower_{window}"] = df[f"bb_middle_{window}"] - (
            2 * df[f"bb_std_{window}"]
        )
        df[f"bb_width_{window}"] = (
            df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
        ) / df[f"bb_middle_{window}"]
        df[f"bb_position_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (
            df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
        )

    # Feature 3: RSI (Relative Strength Index)
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()

        # Avoid division by zero
        loss = loss.replace(0, 0.00001)

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = calculate_rsi(df["close"], 14)

    # RSI derived features
    df["rsi_overbought"] = df["rsi_14"] > 70
    df["rsi_oversold"] = df["rsi_14"] < 30

    # Feature 4: MACD (Moving Average Convergence Divergence)
    df["macd_line"] = (
        df["close"].ewm(span=12, adjust=False).mean()
        - df["close"].ewm(span=26, adjust=False).mean()
    )
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
    df["macd_crossover"] = (df["macd_line"] > df["macd_signal"]).astype(int)

    # Feature 5: Price patterns and momentum
    # Price compared to recent ranges
    df["price_vs_20d_high"] = df["close"] / df["high"].rolling(window=20).max()
    df["price_vs_20d_low"] = df["close"] / df["low"].rolling(window=20).min()

    # Momentum indicators
    df["momentum_5d"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_10d"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20d"] = df["close"] / df["close"].shift(20) - 1

    # Average True Range (ATR) - volatility indicator
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["atr_14"] = df["tr"].rolling(window=14).mean()

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include=bool).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Drop NaN values from calculations
    df.dropna(inplace=True)

    logger.info(f"Technical indicators added, shape: {df.shape}")
    return df


def prepare_features(
    df, target_type="next_day_direction", dim_reduction="pca", n_components=10
):
    """Prepare features and target for model training with dimensionality reduction"""
    logger.info(
        f"Preparing features for {target_type} prediction using {dim_reduction}..."
    )

    # Select target
    if target_type == "next_day_direction":
        y = df["next_day_direction"]
    elif target_type == "market_regime":
        y = df["market_regime"]
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Exclude target columns and other non-feature columns
    exclude_cols = [
        "next_day_direction",
        "market_regime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply dimensionality reduction
    if dim_reduction == "pca":
        # Principal Component Analysis
        reducer = PCA(n_components=min(n_components, X_train_scaled.shape[1]))
        X_train_reduced = reducer.fit_transform(X_train_scaled)
        X_test_reduced = reducer.transform(X_test_scaled)

        # Get explained variance
        explained_variance = reducer.explained_variance_ratio_.cumsum()
        logger.info(
            f"PCA explained variance with {n_components} components: {explained_variance[-1]:.2f}"
        )

    elif dim_reduction == "select_k_best":
        # Feature selection using ANOVA F-statistic
        reducer = SelectKBest(f_classif, k=min(n_components, X_train_scaled.shape[1]))
        X_train_reduced = reducer.fit_transform(X_train_scaled, y_train)
        X_test_reduced = reducer.transform(X_test_scaled)

        # Get selected feature indices
        selected_feature_indices = reducer.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_feature_indices]
        logger.info(f"Selected features: {selected_features}")

    else:
        # No dimensionality reduction
        X_train_reduced = X_train_scaled
        X_test_reduced = X_test_scaled
        reducer = None

    logger.info(f"Features prepared, training set shape: {X_train_reduced.shape}")

    return (
        X_train_reduced,
        X_test_reduced,
        y_train,
        y_test,
        scaler,
        reducer,
        feature_cols,
    )


def train_model(
    X_train,
    y_train,
    model_type="rf",
    target_type="next_day_direction",
    tune_hyperparams=False,
):
    """Train a classification model with optional hyperparameter tuning"""
    logger.info(f"Training {model_type} model for {target_type}...")

    if model_type == "rf":
        model = RandomForestClassifier(random_state=42)

        if tune_hyperparams:
            # Define hyperparameter grid
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

            # Perform grid search
            logger.info("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring="accuracy", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Get best model
            model = grid_search.best_estimator_
            logger.info(f"Best hyperparameters: {grid_search.best_params_}")

        else:
            # Default hyperparameters
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            )
            model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info("Model training complete")
    return model


def evaluate_model(model, X_test, y_test, target_type="next_day_direction"):
    """Evaluate the model and display metrics"""
    logger.info(f"Evaluating model for {target_type}...")

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nResults for {target_type} prediction:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))

    # Label mapping for better visualization
    if target_type == "next_day_direction":
        labels = ["Down", "Up"]
    elif target_type == "market_regime":
        labels = ["Bear", "Sideways", "Bull"]
    else:
        labels = [str(i) for i in range(len(set(y_test)))]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {target_type}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    # Save the confusion matrix
    os.makedirs(f"models/Classification/{target_type}", exist_ok=True)
    plt.savefig(f"models/Classification/{target_type}/confusion_matrix.png")
    plt.close()

    return predictions, accuracy


def plot_feature_importance(
    model,
    feature_cols,
    reducer=None,
    dim_reduction="none",
    target_type="next_day_direction",
):
    """Plot feature importance from the model"""
    logger.info("Plotting feature importance...")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        if dim_reduction == "pca":
            # For PCA, we can't directly map to original features
            plt.figure(figsize=(10, 6))
            plt.title("PCA Component Importance")
            plt.bar(range(len(importances)), importances)
            plt.xlabel("Principal Component")
            plt.ylabel("Importance")

        elif dim_reduction == "select_k_best":
            # For feature selection, we can map to selected features
            selected_features = [
                feature_cols[i] for i in reducer.get_support(indices=True)
            ]

            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)),
                [selected_features[i] for i in indices],
                rotation=90,
            )
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.tight_layout()

        else:
            # No dimensionality reduction, all features
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(14, 10))
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)), [feature_cols[i] for i in indices], rotation=90
            )
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.tight_layout()

        # Save the feature importance plot
        os.makedirs(f"models/Classification/{target_type}", exist_ok=True)
        plt.savefig(f"models/Classification/{target_type}/feature_importance.png")
        plt.close()
    else:
        logger.warning("Model doesn't have feature_importances_ attribute")


def save_model(model, scaler, reducer, feature_cols, target_type, dim_reduction):
    """Save the model, scaler, reducer and feature columns"""
    logger.info(f"Saving model for {target_type}...")

    model_dir = f"models/Classification/{target_type}"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save reducer if used
    if reducer is not None:
        with open(f"{model_dir}/reducer.pkl", "wb") as f:
            pickle.dump(reducer, f)

    # Save feature columns
    with open(f"{model_dir}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    # Save configuration
    config = {
        "target_type": target_type,
        "dim_reduction": dim_reduction,
        "feature_count": len(feature_cols),
    }
    with open(f"{model_dir}/config.pkl", "wb") as f:
        pickle.dump(config, f)

    logger.info(f"Model saved to {model_dir}")


def main():
    try:
        # Load and prepare data
        df = load_data()
        df = add_technical_indicators(df)

        # Configuration options
        configurations = [
            {
                "target_type": "next_day_direction",
                "dim_reduction": "pca",
                "n_components": 10,
            },
            {
                "target_type": "next_day_direction",
                "dim_reduction": "select_k_best",
                "n_components": 10,
            },
            {
                "target_type": "market_regime",
                "dim_reduction": "pca",
                "n_components": 10,
            },
            {
                "target_type": "market_regime",
                "dim_reduction": "select_k_best",
                "n_components": 10,
            },
        ]

        results = {}

        for config in configurations:
            target_type = config["target_type"]
            dim_reduction = config["dim_reduction"]
            n_components = config["n_components"]

            # Prepare features with dimensionality reduction
            X_train, X_test, y_train, y_test, scaler, reducer, feature_cols = (
                prepare_features(df, target_type, dim_reduction, n_components)
            )

            # Train model
            model = train_model(
                X_train,
                y_train,
                model_type="rf",
                target_type=target_type,
                tune_hyperparams=True,
            )

            # Evaluate model
            predictions, accuracy = evaluate_model(model, X_test, y_test, target_type)

            # Plot feature importance
            plot_feature_importance(
                model, feature_cols, reducer, dim_reduction, target_type
            )

            # Save model and artifacts
            save_model(model, scaler, reducer, feature_cols, target_type, dim_reduction)

            # Store results
            results[f"{target_type}_{dim_reduction}"] = {
                "accuracy": accuracy,
                "predictions": predictions,
                "model": model,
            }

            logger.info(f"Completed {target_type} prediction using {dim_reduction}")

        # Compare results
        print("\nModel Comparison:")
        for key, value in results.items():
            print(f"{key}: Accuracy = {value['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
