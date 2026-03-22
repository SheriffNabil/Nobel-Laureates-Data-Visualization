"""
ml_models.py — Machine Learning & Deep Learning for Nobel Laureate Data
Includes: K-Means clustering, Random Forest classification, NLP topic modeling,
and PyTorch LSTM time-series forecasting.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, silhouette_score
import torch
import torch.nn as nn


# ===========================================================================
# 1. K-Means Clustering
# ===========================================================================

def kmeans_clustering(df: pd.DataFrame, n_clusters: int = 5) -> dict:
    """
    Cluster laureates by (award_year, age_at_award, category_encoded).
    Returns cluster labels and centroids for visualization.
    """
    persons = df[~df["is_org"]].dropna(subset=["age_at_award", "award_year"]).copy()

    # One-hot encode category
    dummies = pd.get_dummies(persons["category"], prefix="cat")
    features_df = pd.concat([persons[["award_year", "age_at_award"]], dummies], axis=1)

    features = features_df.values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Find optimal k using silhouette score
    silhouette_scores = {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_scaled)
        silhouette_scores[k] = round(silhouette_score(features_scaled, labels), 4)

    # Fit final model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    persons["cluster"] = kmeans.fit_predict(features_scaled)

    # Cluster profiles
    profiles = (
        persons.groupby("cluster")
        .agg(
            count=("id", "count"),
            avg_year=("award_year", "mean"),
            avg_age=("age_at_award", "mean"),
            top_category=("category", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
            top_country=("birth_country", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
        )
        .round(1)
    )

    return {
        "data": persons[["name", "award_year", "age_at_award", "category", "cluster", "birth_country"]],
        "profiles": profiles,
        "silhouette_scores": silhouette_scores,
        "best_silhouette": silhouette_scores.get(n_clusters, 0),
        "n_clusters": n_clusters,
    }


# ===========================================================================
# 2. Random Forest Classification
# ===========================================================================

def random_forest_category(df: pd.DataFrame) -> dict:
    """
    Predict prize category from (gender, birth_continent, decade, age_at_award).
    Returns feature importances and cross-validation accuracy.
    """
    persons = df[~df["is_org"]].dropna(
        subset=["age_at_award", "birth_continent", "decade"]
    ).copy()

    # Target encoding
    le_category = LabelEncoder()
    y = le_category.fit_transform(persons["category"])

    # One-hot encode features
    X_df = pd.get_dummies(persons[["gender", "birth_continent"]], drop_first=True)
    X_df["decade"] = persons["decade"]
    X_df["age_at_award"] = persons["age_at_award"]
    
    feature_labels = list(X_df.columns)
    X = X_df.values

    # Train model with balanced class weights
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight="balanced")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

    rf.fit(X, y)
    importances = rf.feature_importances_
    importance_dict = dict(zip(feature_labels, [round(x, 4) for x in importances]))
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    return {
        "feature_importances": importance_dict,
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "cv_scores": [round(x, 4) for x in cv_scores.tolist()],
        "categories": le_category.classes_.tolist(),
        "n_samples": len(y),
    }


# ===========================================================================
# 3. NLP Topic Modeling on Motivations
# ===========================================================================

def topic_modeling(df: pd.DataFrame, n_topics: int = 8) -> dict:
    """
    TF-IDF + Truncated SVD topic modeling on prize motivations.
    Extracts latent research themes across categories.
    """
    motivations = df[df["motivation"].str.len() > 10].copy()

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf_matrix = tfidf.fit_transform(motivations["motivation"])

    # Truncated SVD (LSA)
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    topic_matrix = svd.fit_transform(tfidf_matrix)

    # Feature names
    feature_names = tfidf.get_feature_names_out()

    # Extract top words per topic
    topics = {}
    for i, component in enumerate(svd.components_):
        top_indices = component.argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        top_weights = [round(float(component[j]), 4) for j in top_indices]
        topics[f"Topic {i + 1}"] = {
            "words": top_words,
            "weights": top_weights,
        }

    # Dominant topic per category
    motivations["dominant_topic"] = topic_matrix.argmax(axis=1)
    topic_by_category = pd.crosstab(
        motivations["category"], motivations["dominant_topic"]
    )

    explained_var = round(svd.explained_variance_ratio_.sum() * 100, 1)

    return {
        "topics": topics,
        "topic_by_category": topic_by_category,
        "explained_variance": explained_var,
        "n_documents": len(motivations),
        "topic_matrix": topic_matrix,
        "names": motivations["name"].tolist(),
        "categories": motivations["category"].tolist(),
    }


# ===========================================================================
# 4. PyTorch LSTM Forecasting
# ===========================================================================

class PrizeLSTM(nn.Module):
    """LSTM model for time-series prize count forecasting."""

    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def lstm_forecast(df: pd.DataFrame, forecast_years: int = 10, seq_length: int = 10) -> dict:
    """
    Train an LSTM to forecast total Nobel prizes per year.
    """
    # Prepare time series: prizes per year
    yearly = df.groupby("award_year").size().reset_index(name="count")
    yearly = yearly.sort_values("award_year")

    years = yearly["award_year"].values
    counts = yearly["count"].values.astype(np.float32)

    # Normalize
    mean_count = counts.mean()
    std_count = counts.std()
    counts_norm = (counts - mean_count) / std_count

    # Create sequences
    X_list, y_list = [], []
    for i in range(len(counts_norm) - seq_length):
        X_list.append(counts_norm[i: i + seq_length])
        y_list.append(counts_norm[i + seq_length])

    X = torch.FloatTensor(np.array(X_list)).unsqueeze(-1)
    y = torch.FloatTensor(np.array(y_list)).unsqueeze(-1)

    # Train
    model = PrizeLSTM(input_size=1, hidden_size=16, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Predict future
    model.eval()
    with torch.no_grad():
        # First, get fitted values
        fitted = model(X).numpy().ravel() * std_count + mean_count

        # Forecast
        last_seq = counts_norm[-seq_length:].tolist()
        forecasts = []
        for _ in range(forecast_years):
            inp = torch.FloatTensor([last_seq[-seq_length:]]).unsqueeze(-1)
            pred = model(inp).item()
            forecasts.append(pred * std_count + mean_count)
            last_seq.append(pred)

    last_year = int(years[-1])
    forecast_years_list = list(range(last_year + 1, last_year + forecast_years + 1))

    return {
        "historical_years": years[seq_length:].tolist(),
        "historical_actual": counts[seq_length:].tolist(),
        "historical_fitted": fitted.tolist(),
        "forecast_years": forecast_years_list,
        "forecast_values": [round(x, 1) for x in forecasts],
        "training_loss": [round(x, 4) for x in losses],
        "final_loss": round(losses[-1], 6),
    }


# ===========================================================================
# Main entry point
# ===========================================================================

def run_ml(df: pd.DataFrame) -> dict:
    """Run all ML & DL models and return results."""
    print("  🔹 Running K-Means clustering...")
    clustering = kmeans_clustering(df)

    print("  🔹 Running Random Forest classification...")
    rf_results = random_forest_category(df)

    print("  🔹 Running NLP topic modeling...")
    nlp_results = topic_modeling(df)

    print("  🔹 Training LSTM forecasting model...")
    forecast = lstm_forecast(df)

    return {
        "clustering": clustering,
        "random_forest": rf_results,
        "topic_modeling": nlp_results,
        "lstm_forecast": forecast,
    }


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    results = run_ml(df)

    print("\n" + "=" * 60)
    print("🤖 ML & DEEP LEARNING RESULTS")
    print("=" * 60)

    print(f"\n--- K-Means Clustering ({results['clustering']['n_clusters']} clusters) ---")
    print(f"  Silhouette: {results['clustering']['best_silhouette']}")
    print(results["clustering"]["profiles"])

    print("\n--- Random Forest ---")
    print(f"  CV Accuracy: {results['random_forest']['cv_accuracy_mean']:.1%} ± {results['random_forest']['cv_accuracy_std']:.1%}")
    print(f"  Feature importances: {results['random_forest']['feature_importances']}")

    print(f"\n--- NLP Topics ({results['topic_modeling']['explained_variance']}% variance) ---")
    for name, topic in list(results["topic_modeling"]["topics"].items())[:3]:
        print(f"  {name}: {', '.join(topic['words'][:5])}")

    print(f"\n--- LSTM Forecast (final loss: {results['lstm_forecast']['final_loss']}) ---")
    for y, v in zip(results["lstm_forecast"]["forecast_years"], results["lstm_forecast"]["forecast_values"]):
        print(f"  {y}: {v:.1f} prizes")
