"""
visualizations.py — Plotly Interactive Visualizations for Nobel Laureate Data
MiroFish-style white theme: white bg, black text, monospace, orange accents, thin borders.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── MiroFish Color Palette — monochrome with orange accent ────────────────────

COLORS = {
    "Physics": "#000000",
    "Chemistry": "#FF4500",
    "Physiology or Medicine": "#666666",
    "Literature": "#999999",
    "Peace": "#333333",
    "Economic Sciences": "#AAAAAA",
}

CATEGORY_COLORS_SEQ = ["#000000", "#FF4500", "#666666", "#999999", "#333333", "#AAAAAA"]

CATEGORY_ORDER = list(COLORS.keys())

TEMPLATE = "plotly_white"
BG_COLOR = "#FFFFFF"
PAPER_COLOR = "#FFFFFF"
FONT_COLOR = "#000000"
GRID_COLOR = "#EAEAEA"
ACCENT = "#FF4500"
GRAY_TEXT = "#666666"
BORDER = "#E5E5E5"


def _base_layout(fig, title="", height=500):
    """Apply MiroFish white theme to a figure."""
    fig.update_layout(
        template=TEMPLATE,
        title=None,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=PAPER_COLOR,
        font=dict(color=FONT_COLOR, family="Space Grotesk, system-ui, sans-serif", size=12),
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(family="JetBrains Mono, monospace", size=11),
        ),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=BORDER, linewidth=1)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=BORDER, linewidth=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ══════════════════════════════════════════════════════════════════════════════




def fig_sunburst(df):
    """Stacked bar: prizes by continent, colored by category."""
    sb = df[df["birth_continent"].notna()].copy()
    ct = sb.groupby(["birth_continent", "category"]).size().reset_index(name="count")

    # Order continents by total prizes
    continent_order = ct.groupby("birth_continent")["count"].sum().sort_values(ascending=True).index.tolist()

    fig = go.Figure()
    for cat in CATEGORY_ORDER:
        cat_data = ct[ct["category"] == cat]
        fig.add_trace(go.Bar(
            y=cat_data["birth_continent"],
            x=cat_data["count"],
            name=cat,
            orientation="h",
            marker_color=COLORS.get(cat, "#888"),
            marker_line=dict(width=0.5, color="#FFFFFF"),
        ))
    _base_layout(fig, "Prizes by Continent & Category", 420)
    fig.update_layout(
        barmode="stack",
        yaxis=dict(categoryorder="array", categoryarray=continent_order),
        xaxis_title="Number of Prizes", yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
    )
    return fig


def fig_timeline_cumulative(df):
    """Cumulative laureate count over time."""
    yearly = df.groupby("award_year").size().reset_index(name="count")
    yearly = yearly.sort_values("award_year")
    yearly["cumulative"] = yearly["count"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["award_year"], y=yearly["cumulative"],
        mode="lines", fill="tozeroy",
        line=dict(color="#000000", width=2),
        fillcolor="rgba(0, 0, 0, 0.04)",
        name="Cumulative Count",
    ))
    fig.add_trace(go.Bar(
        x=yearly["award_year"], y=yearly["count"],
        name="Yearly Count", marker_color="rgba(255, 69, 0, 0.4)",
        yaxis="y2",
    ))
    _base_layout(fig, "Cumulative Nobel Prizes Over Time", 420)
    fig.update_layout(
        yaxis=dict(title="Cumulative"),
        yaxis2=dict(title="Per Year", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def fig_category_continent(df):
    """100% stacked horizontal bar: continents on Y-axis, segments by category."""
    ct = pd.crosstab(df["birth_continent"], df["category"])
    # Sort continents by total prizes
    totals = ct.sum(axis=1)
    ct = ct.loc[totals.sort_values().index]
    totals = totals.loc[ct.index]
    # Normalize to percentages
    ct_pct = ct.div(totals, axis=0) * 100

    fig = go.Figure()
    for cat in CATEGORY_ORDER:
        if cat in ct_pct.columns:
            fig.add_trace(go.Bar(
                y=ct_pct.index,
                x=ct_pct[cat],
                name=cat,
                orientation="h",
                marker=dict(color=COLORS.get(cat, "#888"), line=dict(width=0.5, color="#FFFFFF")),
                text=[f"{v:.0f}%" if v >= 5 else "" for v in ct_pct[cat]],
                textposition="inside",
                textfont=dict(family="JetBrains Mono, monospace", size=10, color="#FFFFFF"),
                insidetextanchor="middle",
                customdata=ct[cat],
                hovertemplate="%{y}<br>%{fullData.name}: %{customdata} prizes (%{x:.1f}%)<extra></extra>",
            ))
    _base_layout(fig, "Prizes: Continent × Category", 420)
    fig.update_layout(
        barmode="stack",
        xaxis_title="% of Prizes", yaxis_title="",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DEMOGRAPHICS TAB
# ══════════════════════════════════════════════════════════════════════════════


def fig_gender_decade(df):
    """Stacked bar of gender distribution by decade."""
    persons = df[~df["is_org"] & df["gender"].isin(["male", "female"])].copy()
    ct = pd.crosstab(persons["decade"], persons["gender"])
    fig = go.Figure()
    for g, color in [("female", "#FF4500"), ("male", "#000000")]:
        if g in ct.columns:
            fig.add_trace(go.Bar(
                x=ct.index.astype(str), y=ct[g], name=g.capitalize(),
                marker_color=color,
                marker_line=dict(width=1, color="#FFFFFF"),
            ))
    _base_layout(fig, "Gender Distribution by Decade", 420)
    fig.update_layout(
        barmode="stack", xaxis_title="Decade", yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def fig_gender_pct_trend(df):
    """Female percentage trend over decades."""
    persons = df[~df["is_org"] & df["gender"].isin(["male", "female"])].copy()
    ct = pd.crosstab(persons["decade"], persons["gender"])
    if "female" in ct.columns:
        ct["pct"] = ct["female"] / ct.sum(axis=1) * 100
    else:
        ct["pct"] = 0
    fig = go.Figure(go.Scatter(
        x=ct.index.astype(str), y=ct["pct"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=8, color=ACCENT, line=dict(width=1, color="#000000")),
        fill="tozeroy",
        fillcolor="rgba(255, 69, 0, 0.06)",
    ))
    _base_layout(fig, "Female Laureate % Over Decades", 380)
    fig.update_layout(yaxis_title="Female %", xaxis_title="Decade", showlegend=False)
    return fig


def fig_age_violin(df):
    """Violin/box plots of age at award by category."""
    persons = df[~df["is_org"]].dropna(subset=["age_at_award"]).copy()
    fig = px.violin(
        persons, x="category", y="age_at_award", color="category",
        color_discrete_map=COLORS, box=True, points="outliers",
        category_orders={"category": CATEGORY_ORDER},
    )
    _base_layout(fig, "Age at Award Distribution by Category", 480)
    fig.update_layout(xaxis_title="", yaxis_title="Age at Award", showlegend=False)
    return fig


def fig_age_scatter(df, category=None):
    """Scatter: age vs year, colored by category."""
    persons = df[~df["is_org"]].dropna(subset=["age_at_award"]).copy()
    
    if category and category != "All Categories":
        persons = persons[persons["category"] == category]
        
    fig = px.scatter(
        persons, x="award_year", y="age_at_award",
        color="category", color_discrete_map=COLORS,
        hover_name="name", hover_data=["birth_country"],
        opacity=0.65, size_max=8,
        category_orders={"category": CATEGORY_ORDER},
    )
    
    title = "Age at Award Over Time"
    _base_layout(fig, title, 480)
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color="#FFFFFF")))
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Age at Award",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def fig_lifespan_box(df):
    """Lifespan box plots by category."""
    deceased = df[~df["is_org"] & df["lifespan"].notna()].copy()
    fig = px.box(
        deceased, x="category", y="lifespan", color="category",
        color_discrete_map=COLORS,
        category_orders={"category": CATEGORY_ORDER},
    )
    _base_layout(fig, "Lifespan Distribution by Category", 420)
    fig.update_layout(xaxis_title="", yaxis_title="Lifespan (years)", showlegend=False)
    return fig


def fig_age_histogram(df):
    """Histogram of age at award."""
    persons = df[~df["is_org"]].dropna(subset=["age_at_award"]).copy()
    fig = px.histogram(
        persons, x="age_at_award", nbins=40, color="gender",
        color_discrete_map={"male": "#000000", "female": "#FF4500"},
        marginal="rug", opacity=0.7,
    )
    _base_layout(fig, "Age at Award Distribution", 420)
    fig.update_layout(
        xaxis_title="Age at Award", yaxis_title="Count", barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TRENDS TAB
# ══════════════════════════════════════════════════════════════════════════════


def fig_category_decade_heatmap(df):
    """Heatmap: category × decade."""
    ct = pd.crosstab(df["category"], df["decade"])
    fig = px.imshow(
        ct, color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#FFCCBC"], [1, "#FF4500"]],
        labels=dict(x="Decade", y="Category", color="Count"),
        aspect="auto",
    )
    _base_layout(fig, "Prize Heatmap: Category × Decade", 420)
    return fig


def fig_prize_amount_trend(df):
    """Prize amount trends over time."""
    amounts = df.dropna(subset=["prize_amount_adjusted"]).copy()
    yearly = amounts.groupby("award_year").agg(
        avg_amount=("prize_amount_adjusted", "mean"),
        max_amount=("prize_amount_adjusted", "max"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["award_year"], y=yearly["avg_amount"],
        mode="lines+markers", name="Avg Prize (adjusted)",
        line=dict(color="#000000", width=2),
        marker=dict(size=4, color="#000000"),
    ))
    fig.add_trace(go.Scatter(
        x=yearly["award_year"], y=yearly["max_amount"],
        mode="lines", name="Max Prize (adjusted)",
        line=dict(color=ACCENT, width=1, dash="dash"),
    ))
    _base_layout(fig, "Prize Amount Trends (Inflation Adjusted)", 420)
    fig.update_layout(
        yaxis_title="SEK", xaxis_title="Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def fig_sankey(df):
    """Sankey: Continent → Category flow."""
    flow = df[df["birth_continent"].notna()].copy()
    flow = flow.groupby(["birth_continent", "category"]).size().reset_index(name="count")

    continents = flow["birth_continent"].unique().tolist()
    categories = flow["category"].unique().tolist()
    all_labels = continents + categories

    source = [continents.index(c) for c in flow["birth_continent"]]
    target = [len(continents) + categories.index(c) for c in flow["category"]]

    # Monochrome node colors
    cont_grays = ["#000000", "#333333", "#555555", "#777777", "#999999", "#AAAAAA", "#BBBBBB"]
    cat_colors = [COLORS.get(c, "#888") for c in categories]
    node_colors = cont_grays[:len(continents)] + cat_colors

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_labels,
            color=node_colors,
            line=dict(width=1, color="#E5E5E5"),
        ),
        link=dict(
            source=source, target=target,
            value=flow["count"].tolist(),
            color=["rgba(0,0,0,0.08)"] * len(source),
        ),
    ))
    _base_layout(fig, "Continent → Category Flow", 500)
    return fig


def fig_treemap(df):
    """Treemap: Country → Category."""
    tm = df[df["birth_country"].notna()].copy()
    fig = px.treemap(
        tm, path=["birth_continent", "birth_country", "category"],
        color="category", color_discrete_map=COLORS,
    )
    _base_layout(fig, "Treemap: Region → Country → Category", 550)
    return fig


def fig_category_trends_line(df):
    """Line chart of prizes per category over decades."""
    ct = pd.crosstab(df["decade"], df["category"])
    grays = ["#000000", "#FF4500", "#555555", "#888888", "#333333", "#AAAAAA"]
    fig = go.Figure()
    for i, cat in enumerate(CATEGORY_ORDER):
        if cat in ct.columns:
            fig.add_trace(go.Scatter(
                x=ct.index.astype(str), y=ct[cat],
                mode="lines+markers", name=cat,
                line=dict(color=grays[i % len(grays)], width=2),
                marker=dict(size=6),
            ))
    _base_layout(fig, "Category Trends Over Decades", 420)
    fig.update_layout(
        xaxis_title="Decade", yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION TAB
# ══════════════════════════════════════════════════════════════════════════════


def fig_regression_scatter(regression_result, df):
    """Age regression with trend line overlay."""
    preds = regression_result["predictions"]
    persons = df[~df["is_org"]].dropna(subset=["age_at_award"]).copy()

    fig = go.Figure()
    for cat in CATEGORY_ORDER:
        cat_data = persons[persons["category"] == cat]
        fig.add_trace(go.Scatter(
            x=cat_data["award_year"], y=cat_data["age_at_award"],
            mode="markers", name=cat,
            marker=dict(size=5, color=COLORS.get(cat, "#888"), opacity=0.45),
        ))

    # Trend line
    years_sorted = sorted(zip(preds["year"], preds["predicted"]))
    fig.add_trace(go.Scatter(
        x=[y for y, _ in years_sorted],
        y=[p for _, p in years_sorted],
        mode="lines", name=f"Trend (slope={regression_result['slope']:.3f})",
        line=dict(color=ACCENT, width=3, dash="dash"),
    ))
    _base_layout(fig, f"Age Trend Regression (R²={regression_result['r_squared']:.4f})", 480)
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Age at Award",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def fig_correlation_heatmap(corr_df):
    """Correlation heatmap."""
    fig = px.imshow(
        corr_df, color_continuous_scale=[[0, "#FF4500"], [0.5, "#FFFFFF"], [1, "#000000"]],
        zmin=-1, zmax=1, aspect="auto",
        text_auto=".2f",
    )
    _base_layout(fig, "Feature Correlations", 400)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ML TAB
# ══════════════════════════════════════════════════════════════════════════════


def fig_clusters(clustering_result):
    """Scatter of K-Means clusters."""
    data = clustering_result["data"]
    cluster_colors = ["#000000", "#FF4500", "#666666", "#999999", "#333333", "#CCCCCC", "#AAAAAA"]
    fig = px.scatter(
        data, x="award_year", y="age_at_award",
        color=data["cluster"].astype(str),
        hover_name="name",
        hover_data=["category", "birth_country"],
        color_discrete_sequence=cluster_colors,
        opacity=0.65,
    )
    _base_layout(fig, f"K-Means Clusters (k={clustering_result['n_clusters']})", 480)
    fig.update_layout(
        xaxis_title="Award Year", yaxis_title="Age at Award",
        legend_title="Cluster",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#FFFFFF")))
    return fig


def fig_silhouette_scores(clustering_result):
    """Silhouette score by k."""
    scores = clustering_result["silhouette_scores"]
    fig = go.Figure(go.Bar(
        x=[str(k) for k in scores.keys()],
        y=list(scores.values()),
        marker=dict(color="#000000", line=dict(width=1, color="#FFFFFF")),
        text=[f"{v:.3f}" for v in scores.values()],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
    ))
    _base_layout(fig, "Silhouette Scores by k", 350)
    fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Silhouette Score")
    return fig


def fig_feature_importances(rf_result):
    """Bar chart of Random Forest feature importances."""
    imp = rf_result["feature_importances"]
    fig = go.Figure(go.Bar(
        x=list(imp.values()),
        y=list(imp.keys()),
        orientation="h",
        marker=dict(color="#000000", line=dict(width=1, color="#FFFFFF")),
        text=[f"{v:.3f}" for v in imp.values()],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
    ))
    _base_layout(fig, f"Demographic Feature Importance (Acc: {rf_result['cv_accuracy_mean']:.1%})", 350)
    fig.update_layout(xaxis_title="Importance", yaxis_title="")
    return fig


def fig_topic_heatmap(topic_result):
    """Heatmap of topic × category distribution."""
    tc = topic_result["topic_by_category"]
    fig = px.imshow(
        tc, color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#FFCCBC"], [1, "#FF4500"]],
        labels=dict(x="Topic", y="Category", color="Count"),
        aspect="auto",
    )
    _base_layout(fig, f"NLP Topic Distribution ({topic_result['explained_variance']}% var explained)", 400)
    return fig


def fig_topic_words(topic_result, topic_idx=0):
    """Top words for a specific topic."""
    topic_name = list(topic_result["topics"].keys())[topic_idx]
    topic_data = topic_result["topics"][topic_name]
    fig = go.Figure(go.Bar(
        x=topic_data["weights"],
        y=topic_data["words"],
        orientation="h",
        marker=dict(color="#000000", line=dict(width=0)),
    ))
    _base_layout(fig, f"{topic_name}: Top Keywords", 380)
    fig.update_layout(xaxis_title="TF-IDF Weight", yaxis_title="")
    return fig


def fig_lstm_forecast(forecast_result):
    """LSTM forecast chart with historical data."""
    fig = go.Figure()

    # Historical actual
    fig.add_trace(go.Scatter(
        x=forecast_result["historical_years"],
        y=forecast_result["historical_actual"],
        mode="lines", name="Historical Actual",
        line=dict(color="#000000", width=2),
    ))

    # Historical fitted
    fig.add_trace(go.Scatter(
        x=forecast_result["historical_years"],
        y=forecast_result["historical_fitted"],
        mode="lines", name="LSTM Fitted",
        line=dict(color="#999999", width=2, dash="dot"),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_result["forecast_years"],
        y=forecast_result["forecast_values"],
        mode="lines+markers", name="LSTM Forecast",
        line=dict(color=ACCENT, width=3),
        marker=dict(size=8, color=ACCENT, line=dict(width=1, color="#000000")),
    ))

    # Dividing line
    last_hist = forecast_result["historical_years"][-1]
    fig.add_vline(x=last_hist, line_dash="dash", line_color="#E5E5E5")

    _base_layout(fig, f"LSTM Prize Count Forecast (Loss: {forecast_result['final_loss']:.4f})", 450)
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Prizes per Year",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
        ),
    )
    return fig


def fig_training_loss(forecast_result):
    """LSTM training loss curve."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(forecast_result["training_loss"]))),
        y=forecast_result["training_loss"],
        mode="lines", line=dict(color="#000000", width=2),
        fill="tozeroy", fillcolor="rgba(0, 0, 0, 0.04)",
    ))
    _base_layout(fig, "LSTM Training Loss", 300)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="MSE Loss", showlegend=False)
    return fig


def fig_multi_laureates(df):
    """Laureates who won more than one Nobel Prize."""
    counts = df.groupby(["id", "name"]).size().reset_index(name="prizes")
    multi = counts[counts["prizes"] > 1].sort_values("prizes", ascending=False)

    if len(multi) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No multi-prize laureates found", showarrow=False, font=dict(size=16))
        _base_layout(fig, "Multi-Prize Laureates", 300)
        return fig

    # Abbreviate long organization names
    _ABBR = {
        "International Committee of the Red Cross": "ICRC",
        "Office of the United Nations High Commissioner for Refugees": "UNHCR",
        "United Nations High Commissioner for Refugees": "UNHCR",
        "Médecins Sans Frontières": "MSF",
        "International Atomic Energy Agency": "IAEA",
        "International Campaign to Abolish Nuclear Weapons": "ICAN",
        "Intergovernmental Panel on Climate Change": "IPCC",
        "European Union": "EU",
        "United Nations": "UN",
        "United Nations Children's Fund": "UNICEF",
        "International Labour Organization": "ILO",
        "Organisation for the Prohibition of Chemical Weapons": "OPCW",
        "Pugwash Conferences on Science and World Affairs": "Pugwash Conf.",
        "American Friends Service Committee": "AFSC",
        "Institut de Droit International": "IDI",
        "International Peace Bureau": "IPB",
    }

    details = df[df["id"].isin(multi["id"])].copy()
    details["display_name"] = details["name"].map(lambda n: _ABBR.get(n, n))

    fig = px.scatter(
        details, x="award_year", y="display_name", color="category",
        color_discrete_map=COLORS, size_max=15,
        hover_data={"name": True, "motivation": True, "display_name": False},
    )
    _base_layout(fig, "Multi-Prize Laureates", 350)
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="#FFFFFF")))
    fig.update_layout(
        xaxis_title="Year", yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3D GLOBE TAB
# ══════════════════════════════════════════════════════════════════════════════

# Country centroids (lat, lon) for the most common Nobel countries
_COUNTRY_COORDS = {
    "USA": (39.8, -98.6), "United Kingdom": (54.0, -2.0), "Germany": (51.2, 10.4),
    "France": (46.6, 2.2), "Sweden": (62.0, 15.0), "Japan": (36.2, 138.3),
    "Russia": (61.5, 105.3), "Switzerland": (46.8, 8.2), "Canada": (56.1, -106.3),
    "Netherlands": (52.1, 5.3), "Italy": (41.9, 12.6), "Austria": (47.5, 13.6),
    "Norway": (60.5, 8.5), "Denmark": (56.3, 9.5), "Poland": (51.9, 19.1),
    "Australia": (-25.3, 133.8), "Belgium": (50.5, 4.5), "India": (20.6, 79.0),
    "Hungary": (47.2, 19.5), "China": (35.9, 104.2), "Spain": (40.5, -3.7),
    "South Africa": (-30.6, 22.9), "Israel": (31.0, 34.9), "Ireland": (53.4, -8.2),
    "Argentina": (-38.4, -63.6), "Finland": (61.9, 25.7), "Czech Republic": (49.8, 15.5),
    "Egypt": (26.8, 30.8), "Portugal": (39.4, -8.2), "Turkey": (39.0, 35.2),
    "Colombia": (4.6, -74.3), "Mexico": (23.6, -102.6), "Brazil": (-14.2, -51.9),
    "Greece": (39.1, 21.8), "Chile": (-35.7, -71.5), "New Zealand": (-40.9, 174.9),
    "Kenya": (-0.02, 37.9), "Nigeria": (9.1, 8.7), "Pakistan": (30.4, 69.3),
    "Bangladesh": (23.7, 90.4), "Myanmar": (21.9, 96.0), "Iran": (32.4, 53.7),
    "Iraq": (33.2, 43.7), "South Korea": (35.9, 127.8), "Taiwan": (23.7, 121.0),
    "Vietnam": (14.1, 108.3), "Thailand": (15.9, 100.5), "Philippines": (12.9, 121.8),
    "Ukraine": (48.4, 31.2), "Romania": (45.9, 25.0), "Croatia": (45.1, 15.2),
    "Lithuania": (55.2, 23.9), "Latvia": (56.9, 24.1), "Estonia": (58.6, 25.0),
    "Iceland": (64.9, -19.0), "Luxembourg": (49.8, 6.1), "Guatemala": (15.8, -90.2),
    "Costa Rica": (9.7, -83.8), "Peru": (-9.2, -75.0), "Venezuela": (6.4, -66.6),
    "Trinidad and Tobago": (10.4, -61.2), "Dominican Republic": (18.7, -70.2),
    "Jamaica": (18.1, -77.3), "Cuba": (21.5, -77.8), "Puerto Rico": (18.2, -66.6),
    "Ghana": (7.9, -1.0), "Tanzania": (-6.4, 34.9), "Ethiopia": (9.1, 40.5),
    "Liberia": (6.4, -9.4), "Algeria": (28.0, 1.7), "Morocco": (31.8, -7.1),
    "Tunisia": (33.9, 9.5), "Yemen": (15.6, 48.5), "Libya": (26.3, 17.2),
    "Palestine": (31.9, 35.2), "Lebanon": (33.9, 35.9), "Syria": (35.0, 38.0),
    "East Timor": (-8.9, 125.7), "Belarus": (53.7, 27.9), "Bosnia and Herzegovina": (43.9, 17.7),
    "North Macedonia": (41.5, 21.7), "Slovakia": (48.7, 19.7), "Slovenia": (46.1, 14.6),
    "Serbia": (44.0, 21.0), "Montenegro": (42.7, 19.4), "Bulgaria": (42.7, 25.5),
    "Albania": (41.2, 20.2), "Moldova": (47.4, 28.4), "Georgia": (42.3, 43.4),
    "Armenia": (40.1, 45.0), "Azerbaijan": (40.1, 47.6), "Uzbekistan": (41.4, 64.6),
    "Kazakhstan": (48.0, 68.0), "Kyrgyzstan": (41.2, 74.8), "Tajikistan": (38.9, 71.3),
    "Turkmenistan": (38.9, 59.6), "Mongolia": (46.9, 103.8), "Nepal": (28.4, 84.1),
    "Sri Lanka": (7.9, 80.8), "Cambodia": (12.6, 105.0), "Laos": (19.9, 102.5),
    "Indonesia": (-0.8, 113.9), "Malaysia": (4.2, 101.9), "Singapore": (1.4, 103.8),
    "Brunei": (4.5, 114.7),
}


def _get_coords(country):
    """Get lat/lon for a country, returning None if not found."""
    return _COUNTRY_COORDS.get(country, None)


def _sphere_xyz(lat, lon, r=1.0):
    """Convert lat/lon to 3D cartesian coordinates on a sphere."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = r * np.cos(lat_r) * np.cos(lon_r)
    y = r * np.cos(lat_r) * np.sin(lon_r)
    z = r * np.sin(lat_r)
    return x, y, z


def _generate_globe_wireframe(r=0.995, n_lat=24, n_lon=36):
    """Generate wireframe lines for a 3D globe."""
    traces = []

    # Latitude lines
    for lat in np.linspace(-80, 80, n_lat):
        lons = np.linspace(-180, 180, 180)
        xs, ys, zs = [], [], []
        for lon in lons:
            x, y, z = _sphere_xyz(lat, lon, r)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(200,200,200,0.25)", width=0.7),
            showlegend=False, hoverinfo="skip",
        ))

    # Longitude lines
    for lon in np.linspace(-180, 180, n_lon):
        lats = np.linspace(-90, 90, 90)
        xs, ys, zs = [], [], []
        for lat in lats:
            x, y, z = _sphere_xyz(lat, lon, r)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(200,200,200,0.25)", width=0.7),
            showlegend=False, hoverinfo="skip",
        ))

    # Equator (thicker)
    lons_eq = np.linspace(-180, 180, 360)
    xs_eq, ys_eq, zs_eq = [], [], []
    for lon in lons_eq:
        x, y, z = _sphere_xyz(0, lon, r)
        xs_eq.append(x)
        ys_eq.append(y)
        zs_eq.append(z)
    traces.append(go.Scatter3d(
        x=xs_eq, y=ys_eq, z=zs_eq, mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=1.2),
        showlegend=False, hoverinfo="skip",
    ))

    return traces


def _globe_layout(fig, title="", height=700):
    """Apply 3D globe layout settings."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=FONT_COLOR, family="JetBrains Mono, Space Grotesk, monospace"),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FAFAFA",
        font=dict(color=FONT_COLOR, family="Space Grotesk, system-ui, sans-serif", size=12),
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        geo=dict(
            projection_type="orthographic",
            showcoastlines=True, coastlinecolor="#CBD5E1", coastlinewidth=1,
            showland=True, landcolor="#FFFFFF",
            showocean=True, oceancolor="#F8FAFC",
            showlakes=True, lakecolor="#F8FAFC",
            showcountries=True, countrycolor="#E2E8F0", countrywidth=0.6,
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            projection_scale=1.02,
            projection_rotation=dict(lon=10, lat=20, roll=0),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(family="JetBrains Mono, monospace", size=10),
            x=0.01, y=0.99,
        ),
    )
    return fig


def fig_globe_3d(df):
    """3D interactive globe with bubble markers for each country."""
    counts = df.groupby("birth_country").agg(
        n_prizes=("id", "size"),
        categories=("category", lambda x: ", ".join(sorted(x.unique()))),
        laureates=("name", lambda x: ", ".join(sorted(x.unique())[:5]) + ("..." if x.nunique() > 5 else "")),
    ).reset_index()

    fig = go.Figure()

    # Country bubbles
    lats, lons, sizes, hover_texts, colors = [], [], [], [], []
    max_prizes = counts["n_prizes"].max()

    for _, row in counts.iterrows():
        coords = _get_coords(row["birth_country"])
        if coords is None:
            continue
        lats.append(coords[0])
        lons.append(coords[1])
        # Scale bubble size: min 6, max 46
        size = 6 + (row["n_prizes"] / max_prizes) * 40
        sizes.append(size)
        hover_texts.append(
            f"<b>{row['birth_country']}</b><br>"
            f"Prizes: {row['n_prizes']}<br>"
            f"Fields: {row['categories']}<br>"
            f"Notable: {row['laureates']}"
        )
        # Color by number of prizes — orange gradient
        intensity = row["n_prizes"] / max_prizes
        if intensity > 0.5:
            colors.append("#FF4500")
        elif intensity > 0.2:
            colors.append("#FF7043")
        elif intensity > 0.05:
            colors.append("#333333")
        else:
            colors.append("#999999")

    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.85,
            line=dict(width=0.5, color="#FFFFFF"),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Countries",
        showlegend=False,
    ))

    # Add highlighted top 5 country labels
    top5 = counts.nlargest(5, "n_prizes")
    for _, row in top5.iterrows():
        coords = _get_coords(row["birth_country"])
        if coords is None:
            continue
        fig.add_trace(go.Scattergeo(
            lon=[coords[1]], lat=[coords[0]],
            mode="text",
            text=[f"{row['birth_country']} ({row['n_prizes']})"],
            textposition="top center",
            textfont=dict(size=10, color="#000000", family="JetBrains Mono, monospace"),
            showlegend=False, hoverinfo="skip",
        ))

    _globe_layout(fig, "Nobel Prize Global Distribution — 3D Globe", 720)
    return fig

