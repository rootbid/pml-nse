import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("clustering_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create output directory
OUTPUT_DIR = "models/Clustering"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load NIFTY data from CSV file"""
    try:
        logger.info("Loading NIFTY data...")
        df = pd.read_csv("data/nifty_data.csv", parse_dates=["date"], index_col="date")
        logger.info(f"Data loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def engineer_features(df):
    """Engineer features for clustering analysis"""
    logger.info("Engineering features for clustering analysis...")

    # Create a copy to avoid modifying the original
    df_features = df.copy()

    # Basic returns and volatility
    df_features["return"] = df_features["close"].pct_change() * 100
    df_features["log_return"] = np.log(
        df_features["close"] / df_features["close"].shift(1)
    )

    # Volatility windows
    for window in [5, 20, 50]:
        df_features[f"volatility_{window}d"] = (
            df_features["return"].rolling(window=window).std()
        )

    # Moving averages
    for window in [5, 20, 50, 200]:
        df_features[f"sma_{window}"] = (
            df_features["close"].rolling(window=window).mean()
        )
        df_features[f"close_to_sma_{window}"] = (
            df_features["close"] / df_features[f"sma_{window}"]
        )

    # Moving average crossovers
    df_features["sma_5_20_ratio"] = df_features["sma_5"] / df_features["sma_20"]
    df_features["sma_20_50_ratio"] = df_features["sma_20"] / df_features["sma_50"]
    df_features["sma_50_200_ratio"] = df_features["sma_50"] / df_features["sma_200"]

    # RSI (Relative Strength Index)
    def calculate_rsi(prices, n=14):
        deltas = np.diff(prices)
        seed = deltas[: n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(n, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n
            rs = up / down if down != 0 else np.inf
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    df_features["rsi_14"] = calculate_rsi(df_features["close"].values, 14)

    # Bollinger Bands
    window = 20
    df_features["bb_middle"] = df_features["close"].rolling(window=window).mean()
    df_features["bb_std"] = df_features["close"].rolling(window=window).std()
    df_features["bb_width"] = (
        df_features["bb_middle"]
        + 2 * df_features["bb_std"]
        - (df_features["bb_middle"] - 2 * df_features["bb_std"])
    ) / df_features["bb_middle"]
    df_features["bb_position"] = (
        df_features["close"] - (df_features["bb_middle"] - 2 * df_features["bb_std"])
    ) / (
        (df_features["bb_middle"] + 2 * df_features["bb_std"])
        - (df_features["bb_middle"] - 2 * df_features["bb_std"])
    )

    # Trend features
    df_features["trend_20d"] = df_features["close"].pct_change(20) * 100
    df_features["trend_50d"] = df_features["close"].pct_change(50) * 100

    # Volume indicators
    df_features["volume_change"] = df_features["volume"].pct_change() * 100
    df_features["volume_ma_ratio"] = (
        df_features["volume"] / df_features["volume"].rolling(window=20).mean()
    )

    # Price rate of change
    for window in [5, 20, 50]:
        df_features[f"roc_{window}d"] = (
            df_features["close"].pct_change(periods=window) * 100
        )

    # Check for NaN values after calculations
    nan_columns = df_features.columns[df_features.isna().any()].tolist()
    if nan_columns:
        logger.warning(f"NaN values found in columns after calculations: {nan_columns}")
        logger.warning(f"NaN counts: {df_features[nan_columns].isna().sum()}")

        # Fill any remaining NaNs with appropriate methods
        # For time series, forward fill is often appropriate
        df_features = df_features.fillna(method="ffill").fillna(method="bfill")

        # For any still remaining NaNs (e.g., at the beginning), fill with zeros or median
        df_features = df_features.fillna(df_features.median()).fillna(0)

    logger.info(f"Feature engineering complete. Final shape: {df_features.shape}")

    return df_features


def select_clustering_features(df):
    """Select relevant features for clustering"""
    logger.info("Selecting features for clustering...")

    features = [
        "volatility_20d",  # Medium-term volatility
        "close_to_sma_50",  # Price relative to medium-term trend
        "close_to_sma_200",  # Price relative to long-term trend
        "sma_20_50_ratio",  # Medium-term trend direction
        "sma_50_200_ratio",  # Long-term trend direction
        "rsi_14",  # Momentum
        "bb_width",  # Volatility measure
        "bb_position",  # Position within volatility bands
        "trend_50d",  # Medium-term trend
        "volume_ma_ratio",  # Volume behavior
        "roc_20d",  # Rate of change
    ]

    X = df[features]
    logger.info(f"Selected {len(features)} features for clustering: {features}")

    return X, features


def normalize_features(X):
    """Normalize features for clustering"""
    logger.info("Normalizing features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def reduce_dimensions(X_scaled, n_components=2):
    """Reduce dimensions with PCA for visualization"""
    logger.info(f"Reducing dimensions with PCA to {n_components} components...")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"Explained variance: {explained_variance}")
    logger.info(f"Total explained variance: {sum(explained_variance):.4f}")

    return X_pca, pca


def determine_optimal_clusters(X_scaled, max_clusters=10):
    """Determine optimal number of clusters using multiple methods"""
    logger.info("Determining optimal number of clusters...")

    # Silhouette scores
    silhouette_scores = []
    calinski_scores = []
    inertia_values = []

    for n_clusters in range(2, max_clusters + 1):
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Calculate scores
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_avg = calinski_harabasz_score(X_scaled, cluster_labels)
        inertia = kmeans.inertia_

        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski_avg)
        inertia_values.append(inertia)

        logger.info(
            f"Clusters: {n_clusters}, Silhouette: {silhouette_avg:.4f}, "
            f"Calinski-Harabasz: {calinski_avg:.4f}, Inertia: {inertia:.4f}"
        )

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Silhouette score
    ax[0].plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
    ax[0].set_title("Silhouette Score")
    ax[0].set_xlabel("Number of clusters")
    ax[0].set_ylabel("Score")
    ax[0].grid(True)

    # Calinski-Harabasz score
    ax[1].plot(range(2, max_clusters + 1), calinski_scores, marker="o")
    ax[1].set_title("Calinski-Harabasz Score")
    ax[1].set_xlabel("Number of clusters")
    ax[1].set_ylabel("Score")
    ax[1].grid(True)

    # Elbow method
    ax[2].plot(range(2, max_clusters + 1), inertia_values, marker="o")
    ax[2].set_title("Elbow Method")
    ax[2].set_xlabel("Number of clusters")
    ax[2].set_ylabel("Inertia")
    ax[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/optimal_clusters.png")
    plt.close()

    # Find optimal clusters
    optimal_silhouette = np.argmax(silhouette_scores) + 2
    optimal_calinski = np.argmax(calinski_scores) + 2

    logger.info(f"Optimal clusters based on Silhouette score: {optimal_silhouette}")
    logger.info(
        f"Optimal clusters based on Calinski-Harabasz score: {optimal_calinski}"
    )

    return optimal_silhouette, optimal_calinski


def kmeans_clustering(X_scaled, n_clusters=3):
    """Perform K-Means clustering"""
    logger.info(f"Performing K-Means clustering with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return kmeans, labels


def gmm_clustering(X_scaled, n_components=3):
    """Perform Gaussian Mixture Model clustering"""
    logger.info(f"Performing GMM clustering with {n_components} components...")

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)

    return gmm, labels, probabilities


def visualize_clusters_2d(X_pca, labels, method="K-Means"):
    """Visualize clusters in 2D using PCA"""
    logger.info(f"Visualizing {method} clusters in 2D...")

    plt.figure(figsize=(12, 8))

    # Create a scatter plot for each cluster
    cluster_colors = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58", "#9C27B0", "#3F51B5"]
    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        plt.scatter(
            X_pca[labels == label, 0],
            X_pca[labels == label, 1],
            s=50,
            c=cluster_colors[i % len(cluster_colors)],
            label=f"Cluster {label}",
        )

    plt.title(f"{method} Clustering Results (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{method.lower().replace(' ', '_')}_clusters_2d.png")
    plt.close()


def visualize_gmm_probabilities(X_pca, probabilities):
    """Visualize GMM cluster probabilities"""
    logger.info("Visualizing GMM cluster probabilities...")

    n_components = probabilities.shape[1]
    plt.figure(figsize=(12, 8))

    # Create a scatter plot with points colored by their highest probability cluster
    dominant_cluster = np.argmax(probabilities, axis=1)

    # Define colors for each cluster
    cluster_colors = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58", "#9C27B0", "#3F51B5"]

    # Size points by the probability of their dominant cluster
    max_probabilities = np.max(probabilities, axis=1)
    sizes = 20 + 100 * max_probabilities

    # Plot each cluster
    for i in range(n_components):
        mask = dominant_cluster == i
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=sizes[mask],
            c=cluster_colors[i % len(cluster_colors)],
            alpha=0.7,
            label=f"Cluster {i}",
        )

    plt.title("GMM Clustering with Probabilities (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/gmm_probabilities.png")
    plt.close()


def analyze_clusters(df, labels, features, method="K-Means"):
    """Analyze and interpret clusters"""
    logger.info(f"Analyzing {method} clusters...")

    # Add cluster labels to the dataframe
    df_analysis = df.copy()
    df_analysis["cluster"] = labels

    # Calculate cluster statistics
    cluster_stats = df_analysis.groupby("cluster")[features].mean()
    logger.info(f"\n{method} Cluster Statistics:\n{cluster_stats}")

    # Create heatmap of cluster centers
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_stats, annot=True, cmap="viridis", fmt=".2f", cbar=True)
    plt.title(f"{method} Cluster Centers", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{method.lower().replace(' ', '_')}_cluster_centers.png")
    plt.close()

    # Interpret market regimes based on cluster characteristics
    market_regimes = []

    for cluster in range(len(cluster_stats)):
        row = cluster_stats.iloc[cluster]

        # Determine market regime based on feature values
        if row["close_to_sma_200"] > 1.0 and row["rsi_14"] > 50:
            if row["volatility_20d"] < cluster_stats["volatility_20d"].median():
                regime = "Strong Bull"
            else:
                regime = "Volatile Bull"
        elif row["close_to_sma_200"] < 1.0 and row["rsi_14"] < 50:
            if row["volatility_20d"] > cluster_stats["volatility_20d"].median():
                regime = "Crisis/Bear"
            else:
                regime = "Weak Bear"
        else:
            if row["bb_width"] < cluster_stats["bb_width"].median():
                regime = "Consolidation/Range"
            else:
                regime = "Transition/Rotation"

        market_regimes.append(regime)

    # Map cluster numbers to regimes
    regime_mapping = {i: regime for i, regime in enumerate(market_regimes)}
    logger.info(f"\n{method} Market Regime Mapping:\n{regime_mapping}")

    # Add regime labels to the dataframe
    df_analysis["market_regime"] = df_analysis["cluster"].map(regime_mapping)

    # Calculate regime statistics
    regime_stats = df_analysis.groupby("market_regime")["return"].agg(
        ["mean", "std", "count"]
    )
    regime_stats["sharpe"] = regime_stats["mean"] / regime_stats["std"]
    logger.info(f"\n{method} Market Regime Return Statistics:\n{regime_stats}")

    # Save regime mapping and statistics
    with open(
        f"{OUTPUT_DIR}/{method.lower().replace(' ', '_')}_regime_analysis.txt", "w"
    ) as f:
        f.write(f"{method} Market Regime Mapping:\n")
        for cluster, regime in regime_mapping.items():
            f.write(f"Cluster {cluster}: {regime}\n")

        f.write("\nCluster Centers:\n")
        f.write(cluster_stats.to_string())

        f.write("\n\nMarket Regime Return Statistics:\n")
        f.write(regime_stats.to_string())

    return df_analysis, regime_mapping


def visualize_market_regimes_over_time(df, labels, regime_mapping, method="K-Means"):
    """Visualize market regimes over time with price"""
    logger.info(f"Visualizing {method} market regimes over time...")

    # Create a copy with labels
    df_plot = df.copy()
    df_plot["cluster"] = labels
    df_plot["regime"] = df_plot["cluster"].map(regime_mapping)

    # Define colors for each regime
    regime_colors = {
        "Strong Bull": "#0F9D58",
        "Volatile Bull": "#7CB342",
        "Consolidation/Range": "#F4B400",
        "Transition/Rotation": "#FF6D00",
        "Weak Bear": "#DB4437",
        "Crisis/Bear": "#9C0000",
    }

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Plot price on the first axis
    ax1.plot(df_plot.index, df_plot["close"], color="black", linewidth=1.5)
    ax1.set_title(f"{method} Market Regimes Over Time", fontsize=16)
    ax1.set_ylabel("NIFTY Close Price", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Highlight regions based on regimes
    unique_regimes = df_plot["regime"].unique()

    # Identify continuous periods of the same regime
    current_regime = None
    start_idx = None

    for idx, row in df_plot.iterrows():
        if row["regime"] != current_regime:
            # End the previous regime period
            if current_regime is not None and start_idx is not None:
                end_idx = idx
                ax1.axvspan(
                    start_idx,
                    end_idx,
                    color=regime_colors.get(current_regime, "#CCCCCC"),
                    alpha=0.3,
                )

            # Start a new regime period
            current_regime = row["regime"]
            start_idx = idx

    # Handle the last regime period
    if current_regime is not None and start_idx is not None:
        ax1.axvspan(
            start_idx,
            df_plot.index[-1],
            color=regime_colors.get(current_regime, "#CCCCCC"),
            alpha=0.3,
        )

    # Add legend for regimes
    legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=regime_colors.get(regime, "#CCCCCC"),
            alpha=0.3,
            label=regime,
        )
        for regime in unique_regimes
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10)

    # Plot regime as a categorical variable on the second axis
    for regime in unique_regimes:
        mask = df_plot["regime"] == regime
        if mask.any():
            ax2.scatter(
                df_plot.index[mask],
                [1] * mask.sum(),
                c=[regime_colors.get(regime, "#CCCCCC")],
                label=regime,
                s=50,
                alpha=0.8,
            )

    ax2.set_yticks([])
    ax2.set_ylabel("Regime", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{method.lower().replace(' ', '_')}_regimes_over_time.png"
    )
    plt.close()


def analyze_regime_transitions(df_analysis, method="K-Means"):
    """Analyze transitions between regimes"""
    logger.info(f"Analyzing {method} regime transitions...")

    # Create transition matrix
    regime_transitions = df_analysis["market_regime"].value_counts().to_dict()

    # Calculate average duration of each regime
    regime_durations = []
    current_regime = None
    current_duration = 0

    for regime in df_analysis["market_regime"]:
        if regime == current_regime:
            current_duration += 1
        else:
            if current_regime is not None:
                regime_durations.append(
                    {"regime": current_regime, "duration": current_duration}
                )
            current_regime = regime
            current_duration = 1

    # Add the last regime
    if current_regime is not None:
        regime_durations.append(
            {"regime": current_regime, "duration": current_duration}
        )

    # Convert to DataFrame and calculate statistics
    durations_df = pd.DataFrame(regime_durations)
    duration_stats = durations_df.groupby("regime")["duration"].agg(
        ["mean", "median", "min", "max", "count"]
    )

    logger.info(f"\n{method} Regime Duration Statistics (days):\n{duration_stats}")

    # Save transition analysis
    with open(
        f"{OUTPUT_DIR}/{method.lower().replace(' ', '_')}_transition_analysis.txt", "w"
    ) as f:
        f.write(f"{method} Regime Frequency:\n")
        for regime, count in regime_transitions.items():
            f.write(f"{regime}: {count} days ({count / len(df_analysis) * 100:.2f}%)\n")

        f.write("\nRegime Duration Statistics (days):\n")
        f.write(duration_stats.to_string())

    return regime_transitions, duration_stats


def compare_clustering_methods(df_kmeans, df_gmm):
    """Compare KMeans and GMM clustering results"""
    logger.info("Comparing clustering methods...")

    comparison = pd.crosstab(
        df_kmeans["market_regime"],
        df_gmm["market_regime"],
        rownames=["KMeans Regime"],
        colnames=["GMM Regime"],
    )

    logger.info(f"\nClustering Method Comparison:\n{comparison}")

    # Create heatmap for comparison
    plt.figure(figsize=(12, 8))
    sns.heatmap(comparison, annot=True, cmap="Blues", fmt="d")
    plt.title("Comparison of Market Regime Classifications", fontsize=14)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/clustering_method_comparison.png")
    plt.close()

    # Calculate agreement percentage
    total_agreement = sum(df_kmeans["market_regime"] == df_gmm["market_regime"])
    agreement_pct = total_agreement / len(df_kmeans) * 100

    logger.info(f"Agreement between clustering methods: {agreement_pct:.2f}%")

    return comparison, agreement_pct


def main():
    try:
        # Load data
        df = load_data()

        # Engineer features
        df_features = engineer_features(df)

        # Select features for clustering
        X, features = select_clustering_features(df_features)

        # Normalize features
        X_scaled, scaler = normalize_features(X)

        # Reduce dimensions for visualization
        X_pca, pca = reduce_dimensions(X_scaled, n_components=2)

        # Determine optimal number of clusters
        silhouette_k, calinski_k = determine_optimal_clusters(X_scaled)

        # Use 3 clusters for interpretability as specified
        n_clusters = 3
        logger.info(f"Using {n_clusters} clusters for interpretability")

        # Perform KMeans clustering
        kmeans, kmeans_labels = kmeans_clustering(X_scaled, n_clusters)

        # Perform GMM clustering
        gmm, gmm_labels, gmm_probs = gmm_clustering(X_scaled, n_clusters)

        # Visualize clusters
        visualize_clusters_2d(X_pca, kmeans_labels, "K-Means")
        visualize_clusters_2d(X_pca, gmm_labels, "GMM")
        visualize_gmm_probabilities(X_pca, gmm_probs)

        # Analyze clusters
        df_kmeans, kmeans_regimes = analyze_clusters(
            df_features, kmeans_labels, features, "K-Means"
        )
        df_gmm, gmm_regimes = analyze_clusters(df_features, gmm_labels, features, "GMM")

        # Visualize market regimes over time
        visualize_market_regimes_over_time(
            df_features, kmeans_labels, kmeans_regimes, "K-Means"
        )
        visualize_market_regimes_over_time(df_features, gmm_labels, gmm_regimes, "GMM")

        # Analyze regime transitions
        kmeans_transitions, kmeans_durations = analyze_regime_transitions(
            df_kmeans, "K-Means"
        )
        gmm_transitions, gmm_durations = analyze_regime_transitions(df_gmm, "GMM")

        # Compare clustering methods
        comparison, agreement = compare_clustering_methods(df_kmeans, df_gmm)

        logger.info("Clustering analysis completed successfully")

        # Return results in dictionary (optional)
        results = {
            "kmeans": {
                "model": kmeans,
                "labels": kmeans_labels,
                "regimes": kmeans_regimes,
            },
            "gmm": {
                "model": gmm,
                "labels": gmm_labels,
                "probabilities": gmm_probs,
                "regimes": gmm_regimes,
            },
            "feature_importance": {
                feature: coef for feature, coef in zip(features, pca.components_[0])
            },
            "optimal_clusters": {"silhouette": silhouette_k, "calinski": calinski_k},
            "agreement": agreement,
        }

        return results

    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        raise


if __name__ == "__main__":
    main()
