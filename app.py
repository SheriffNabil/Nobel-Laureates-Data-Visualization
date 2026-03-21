"""
app.py — Nobel Laureates Interactive Dashboard
MiroFish-style white theme with insight buttons on every chart.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH
import json
import pandas as pd

# ── Local modules ──
from data_loader import load_data, get_persons, get_summary_stats
from analysis import run_analysis
from ml_models import run_ml
import visualizations as viz

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODELS — run once on startup
# ══════════════════════════════════════════════════════════════════════════════

print("🚀 Loading data ...")
DF = load_data(use_api=True)
STATS = get_summary_stats(DF)
print("📈 Running statistical analysis ...")
ANALYSIS = run_analysis(DF)
print("🤖 Running ML & DL models ...")
ML = run_ml(DF)
print("✅ All models ready!\n")

CATEGORIES = sorted(DF["category"].unique().tolist())
YEAR_MIN = int(DF["award_year"].min())
YEAR_MAX = int(DF["award_year"].max())

# ══════════════════════════════════════════════════════════════════════════════
# CHART INSIGHTS — plain-English for non-analysts
# ══════════════════════════════════════════════════════════════════════════════

persons = DF[~DF["is_org"]]
top_country = DF.groupby("birth_country").size().idxmax()
top_country_n = int(DF.groupby("birth_country").size().max())
youngest = persons.loc[persons["age_at_award"].idxmin()] if persons["age_at_award"].notna().any() else None
oldest = persons.loc[persons["age_at_award"].idxmax()] if persons["age_at_award"].notna().any() else None

chi = ANALYSIS["chi_squared"]
mw = ANALYSIS["mann_whitney"]
reg = ANALYSIS["age_regression"]
age_stats = ANALYSIS["age_distribution"]
rf = ML["random_forest"]
cl = ML["clustering"]
tp = ML["topic_modeling"]
fc = ML["lstm_forecast"]


def _build_insights():
    """Build a dictionary of chart_id → insight text."""
    return {
        # ── Overview ──
        "choropleth": {
            "title": "What does this map show?",
            "text": (
                f"This world map colors each country by how many Nobel Prize winners were born there. "
                f"The darker the color, the more laureates. "
                f"**{top_country}** leads with **{top_country_n}** prizes, followed by "
                f"the United Kingdom and Germany. "
                f"Notice how prizes are heavily concentrated in North America and Europe — "
                f"most of the developing world has very few winners, highlighting a geographic imbalance "
                f"in global scientific recognition."
            ),
        },
        "timeline": {
            "title": "What does this timeline tell us?",
            "text": (
                f"The line shows the running total of all Nobel Prizes ever awarded, while the bars show "
                f"how many were given each year. The count grows steadily from 1901, but notice the sharp "
                f"acceleration after the 1960s — more categories and shared prizes mean more awards per year. "
                f"The total has now reached **{STATS['total_prizes']}** prizes. "
                f"You can also spot gaps during World War I and II when fewer prizes were awarded."
            ),
        },
        "categories": {
            "title": "Which fields win the most prizes?",
            "text": (
                "This bar chart ranks the six Nobel Prize categories by total number of awards. "
                "Medicine and Physics typically lead because they've been awarded since 1901, "
                "while Economics was only added in 1969 — so it naturally has fewer prizes. "
                "Each bar represents a different field of human achievement recognized by the Nobel Committee."
            ),
        },
        "top_countries": {
            "title": "Which countries produce the most laureates?",
            "text": (
                f"This ranking shows the top 15 countries by number of Nobel laureates born there. "
                f"**{top_country}** dominates with **{top_country_n}** winners. "
                f"The list is dominated by Western nations, reflecting historical advantages in "
                f"research funding, university systems, and scientific infrastructure. "
                f"However, countries like Japan and India are increasingly represented in recent decades."
            ),
        },
        "sunburst": {
            "title": "How do regions, countries, and fields connect?",
            "text": (
                "This sunburst diagram is like a layered pie chart. The inner ring shows continents, "
                "the middle ring breaks down into countries, and the outer ring shows prize categories. "
                "Click on any segment to zoom in. This reveals patterns — for example, European countries "
                "tend to have a more even spread across categories, while some countries concentrate "
                "in specific fields like Physics or Literature."
            ),
        },
        # ── Demographics ──
        "gender_decade": {
            "title": "How has gender balance changed over time?",
            "text": (
                f"Each bar represents a decade, split between male (black) and female (orange) laureates. "
                f"For most of Nobel history, prizes went almost exclusively to men. "
                f"Female representation has been slowly increasing — from near zero in the early 1900s "
                f"to about **{STATS['female_pct']}%** overall. "
                f"The 2010s and 2020s show the highest numbers of female laureates ever, "
                f"but there's still a significant gap."
            ),
        },
        "gender_trend": {
            "title": "Is the gender gap closing?",
            "text": (
                f"This line tracks the percentage of female winners in each decade. "
                f"The overall average is only **{STATS['female_pct']}%** — meaning roughly 1 in 15 "
                f"Nobel Prize winners is a woman. "
                f"The upward trend in recent decades is encouraging, but progress has been uneven. "
                f"Peace and Literature tend to have more female winners than Physics or Economics."
            ),
        },
        "age_scatter": {
            "title": "At what age do people win Nobel Prizes?",
            "text": (
                f"Each dot represents one laureate, plotted by the year they won (horizontal) "
                f"and their age at the time (vertical). Colors represent different categories. "
                f"The youngest winner was **{youngest['name'] if youngest is not None else 'N/A'}** "
                f"at age **{int(youngest['age_at_award']) if youngest is not None else 'N/A'}**, "
                f"while the oldest was **{oldest['name'] if oldest is not None else 'N/A'}** "
                f"at age **{int(oldest['age_at_award']) if oldest is not None else 'N/A'}**. "
                f"Notice how the dots trend upward over time — laureates are getting older, "
                f"likely because science has become more complex and careers take longer to peak."
            ),
        },
        "age_violin": {
            "title": "How does winning age differ by field?",
            "text": (
                f"These violin plots show the distribution of ages when prizes are awarded, "
                f"one per category. The wider the shape, the more winners at that age. "
                f"Physics laureates tend to be younger (many breakthrough discoveries happen early), "
                f"while Peace and Literature winners are typically older — recognition in those fields "
                f"often comes after decades of work. The average age across all categories is "
                f"**{STATS['avg_age']} years**."
            ),
        },
        "age_histogram": {
            "title": "What's the most common winning age?",
            "text": (
                f"This histogram counts how many laureates won at each age. "
                f"The peak is around age **{age_stats['median']}**, meaning most winners "
                f"are in their late 50s to early 60s. "
                f"The orange bars highlight female winners — they tend to cluster in similar age ranges "
                f"but are far fewer in number. "
                f"Very young winners (under 35) are rare, and so are very old ones (over 85)."
            ),
        },
        "lifespan": {
            "title": "Do Nobel laureates live longer in some fields?",
            "text": (
                "These box plots compare the lifespan of deceased laureates across categories. "
                "The box shows the middle 50% of lifespans, with the line inside marking the median. "
                "Interestingly, laureates across all fields tend to live quite long — often into their "
                "late 70s and 80s. This could reflect the advantages of education, financial stability, "
                "and access to healthcare that often come with a Nobel Prize."
            ),
        },
        # ── Trends & Stats ──
        "hypothesis_tests": {
            "title": "What do these statistical tests mean?",
            "text": (
                "These are formal mathematical tests that check whether patterns in the data are real "
                "or just random chance:\n\n"
                f"**Chi-Squared test**: Checks if gender distribution varies across categories. "
                f"{'It IS significant — men and women are not equally distributed across fields.' if chi['significant'] else 'No significant difference found.'}\n\n"
                f"**Mann-Whitney test**: Compares the typical winning age between Physics and Literature. "
                f"{'The difference IS significant — Physics winners tend to be younger.' if mw['significant'] else 'No significant age difference.'}\n\n"
                f"**Age Distribution**: The Shapiro-Wilk test tells us whether ages follow a bell curve. "
                f"{'They do NOT — the distribution is skewed.' if not age_stats['is_normal'] else 'They do follow a normal distribution.'} "
                f"Average age is **{age_stats['mean']}** with a range from **{age_stats['youngest']}** to **{age_stats['oldest']}**."
            ),
        },
        "regression": {
            "title": "Are Nobel winners getting older over time?",
            "text": (
                f"The dashed orange line shows the mathematical trend. The slope is **{reg['slope']:.3f}**, "
                f"meaning winners are getting about **{abs(reg['slope'] * 10):.1f} years older every decade**. "
                f"{'This trend is statistically significant.' if reg['significant'] else 'However, this trend is not statistically significant.'} "
                f"The R² value of **{reg['r_squared']:.3f}** means the year alone explains about "
                f"**{reg['r_squared']*100:.1f}%** of the variation in age — other factors like field "
                f"and country matter more."
            ),
        },
        "heatmap": {
            "title": "How have categories evolved over decades?",
            "text": (
                "This grid shows prize counts by category (rows) and decade (columns). "
                "Darker cells mean more prizes. You can see that Physics, Chemistry, and Medicine "
                "have been consistently awarded since 1901, while Economics only starts in 1969. "
                "The Peace prize shows more variation — some decades had more conflict-related "
                "recognition than others."
            ),
        },
        "correlations": {
            "title": "How are the numbers related to each other?",
            "text": (
                "This grid shows how strongly different numerical features are connected. "
                "Values range from -1 (opposite relationship) to +1 (move together). "
                "For example, a positive correlation between year and age means older winners in "
                "recent years. Prize amounts increase with year because the Nobel Foundation adjusts "
                "the prize value over time. Lifespan shows weak correlations — meaning how long "
                "laureates live isn't strongly tied to when they won."
            ),
        },
        "prize_amount": {
            "title": "How has the prize money changed?",
            "text": (
                "This chart shows the Nobel Prize monetary value over time, adjusted for inflation. "
                "The black line tracks the average prize amount and the dashed orange line shows the maximum. "
                "The prize was originally 150,000 SEK in 1901 and has grown dramatically. "
                "After adjusting for inflation, the real value has also increased significantly — "
                "today's prize is worth around 11 million SEK (about $1 million USD)."
            ),
        },
        "category_trends": {
            "title": "Which fields are growing fastest?",
            "text": (
                "Each line tracks how many prizes a category awarded per decade. "
                "All categories have generally grown, but some stand out — Medicine and Physics "
                "consistently lead, partly because more prizes are shared among multiple winners. "
                "The Peace prize sometimes has 'blank' years when no suitable candidate is found."
            ),
        },
        # ── ML & AI ──
        "kmeans": {
            "title": "What are these clusters?",
            "text": (
                f"K-Means clustering is a machine learning technique that automatically groups "
                f"similar laureates together. The algorithm found **{cl['n_clusters']} natural groups** "
                f"based on when they won, how old they were, and their field. "
                f"The silhouette score of **{cl['best_silhouette']:.3f}** measures how well-separated "
                f"the groups are (closer to 1.0 = very distinct groups). "
                f"Different colors represent different clusters — you can hover over dots to see "
                f"which laureates belong to which group."
            ),
        },
        "silhouette": {
            "title": "What does the silhouette score mean?",
            "text": (
                "When doing clustering, we need to choose how many groups (k) to create. "
                "The silhouette score helps us decide — higher scores mean the groups are more distinct. "
                "This chart tests different values of k and shows which one creates the best separation. "
                "Values above 0.3 are considered reasonable, above 0.5 is good, and above 0.7 is strong. "
                "The best k doesn't always equal the most clusters — sometimes fewer, cleaner groups "
                "are better than many overlapping ones."
            ),
        },
        "random_forest": {
            "title": "Can we predict someone's prize category?",
            "text": (
                f"A Random Forest is an AI model that learns patterns from data. Here, it tries to "
                f"predict which Nobel category a person wins based on their gender, continent, decade, "
                f"and age. The model's accuracy is **{rf['cv_accuracy_mean']:.1%}** — which is moderate, "
                f"since there are 6 categories to choose from (random guessing would be ~17%). "
                f"**The most important feature is '{list(rf['feature_importances'].keys())[0]}'**, "
                f"meaning it's the most useful clue for predicting category. "
                f"This tells us that when you win matters more than where you're from."
            ),
        },
        "nlp_topics": {
            "title": "What themes appear in prize motivations?",
            "text": (
                f"Natural Language Processing (NLP) analyzed the text of all **{tp['n_documents']}** "
                f"prize motivations to discover hidden themes. It found **{len(tp['topics'])} topics** — "
                f"groups of words that frequently appear together. "
                f"The heatmap shows which topics are most common in each category. "
                f"For example, Physics motivations cluster around words like 'discovery' and 'theoretical', "
                f"while Peace motivations focus on 'efforts' and 'rights'. "
                f"These topics explain **{tp['explained_variance']}%** of the variation in the text."
            ),
        },
        "lstm_forecast": {
            "title": "What does the AI predict for the future?",
            "text": (
                f"This chart uses a neural network called LSTM (Long Short-Term Memory) — "
                f"a type of deep learning AI that learns patterns from sequences of data. "
                f"It was trained on **{YEAR_MAX - YEAR_MIN}+ years** of historical prize counts. "
                f"The black line is the actual history, the gray dotted line shows how well the model "
                f"learned the past, and the orange line projects the next **{len(fc['forecast_years'])} years**. "
                f"The model's final error was **{fc['final_loss']:.4f}** — the lower, the better. "
                f"These forecasts suggest {'relatively stable' if abs(fc['forecast_values'][-1] - fc['forecast_values'][0]) < 3 else 'changing'} "
                f"prize counts in the coming years."
            ),
        },
        "training_loss": {
            "title": "How well did the AI learn?",
            "text": (
                "This shows the AI model's learning progress. The y-axis is the 'error' — how far off "
                "the model's predictions are from reality. As the model trains (x-axis = epochs), "
                "the error should decrease and flatten out. A steep initial drop means the model "
                "quickly learned the main patterns. The final flat portion means it's learned "
                "as much as it can from this data. If it went back up, that would signal 'overfitting' — "
                "memorizing noise rather than learning real patterns."
            ),
        },
        # ── Deep Dive ──
        "sankey": {
            "title": "How do laureates flow from continents to fields?",
            "text": (
                "A Sankey diagram visualizes flow — here it shows how laureates from each continent "
                "distribute across Nobel categories. The width of each ribbon represents the number "
                "of winners. Notice how Europe and North America dominate virtually every category, "
                "while other continents tend to be strongest in Peace. "
                "This visualization makes geographic concentration immediately visible."
            ),
        },
        "treemap": {
            "title": "What's the full breakdown of prizes?",
            "text": (
                "This treemap is a nested box chart — bigger boxes mean more prizes. "
                "The top level is continents, then countries, then categories. "
                "Click on any box to zoom in and explore the breakdown within that region. "
                "It's an intuitive way to see relative proportions — you can immediately spot "
                "that the USA takes up the largest single box, reflecting its dominance."
            ),
        },
        "multi_laureates": {
            "title": "Who won more than once?",
            "text": (
                "Very few people or organizations have won the Nobel Prize more than once. "
                "This chart highlights those exceptional cases. Notable examples include "
                "Marie Curie (Physics 1903 + Chemistry 1911), Linus Pauling (Chemistry 1954 + Peace 1962), "
                "and organizations like the Red Cross and UNHCR which won multiple Peace Prizes. "
                "Winning twice is one of the rarest achievements in human history."
            ),
        },
        # ── Globe ──
        "globe_3d": {
            "title": "What is this 3D globe?",
            "text": (
                f"This is an interactive 3D globe where each bubble represents a country that has produced "
                f"Nobel laureates. The bigger the bubble, the more prizes from that country. "
                f"**{top_country}** has the largest bubble with **{top_country_n}** prizes. "
                f"You can click and drag to rotate the globe, scroll to zoom, and hover over any bubble "
                f"to see the country's name, prize count, fields, and notable laureates. "
                f"Orange bubbles indicate the top producers, while gray ones have fewer prizes. "
                f"This 3D view makes the global concentration of prizes much more visceral than a flat map."
            ),
        },

    }


INSIGHTS = _build_insights()


# ══════════════════════════════════════════════════════════════════════════════
# DASH APP
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    title="Nobel Prize Analytics",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no"}],
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700&display=swap",
    ],
)

# ── Cross-browser CSS fixes (Safari + Chrome consistency) ──
app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
/* Safari/Chrome reset */
*, *::before, *::after { box-sizing: border-box; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
html, body { margin: 0; padding: 0; width: 100%; overflow-x: hidden; }
/* Fix Safari gradient text */
.gradient-text { background: linear-gradient(90deg, #000 0%, #444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; display: inline-block; }
/* Safari button reset */
button { -webkit-appearance: none; appearance: none; }
/* Safari scrollbar match Chrome */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #F5F5F5; }
::-webkit-scrollbar-thumb { background: #CCC; border-radius: 4px; }
/* Safari tab focus ring fix */
.tab-container button:focus { outline: none; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>'''

# ── MiroFish-style ────────────────────────────────────────────────────────────

S = {
    "page": {
        "fontFamily": "'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif",
        "backgroundColor": "#FFFFFF",
        "color": "#000000",
        "minHeight": "100vh",
        "padding": "0",
        "margin": "0",
        "width": "100%",
        "overflowX": "hidden",
    },
    "navbar": {
        "height": "60px",
        "background": "#000000",
        "color": "#FFFFFF",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "0 40px",
    },
    "brand": {
        "fontFamily": "'JetBrains Mono', monospace",
        "fontWeight": "800",
        "fontSize": "1.2rem",
        "letterSpacing": "1px",
        "color": "#FFFFFF",
    },
    "nav_meta": {
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "0.8rem",
        "color": "#999999",
        "display": "flex",
        "alignItems": "center",
        "gap": "20px",
    },
    "hero": {
        "padding": "50px 40px 40px",
        "maxWidth": "1400px",
        "margin": "0 auto",
    },
    "tag_row": {
        "display": "flex",
        "alignItems": "center",
        "gap": "15px",
        "marginBottom": "20px",
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "0.8rem",
    },
    "orange_tag": {
        "background": "#FF4500",
        "color": "#FFFFFF",
        "padding": "4px 10px",
        "fontWeight": "700",
        "letterSpacing": "1px",
        "fontSize": "0.75rem",
        "fontFamily": "'JetBrains Mono', monospace",
    },
    "version_text": {
        "color": "#999999",
        "fontWeight": "500",
        "letterSpacing": "0.5px",
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "0.8rem",
    },
    "main_title": {
        "fontSize": "3.5rem",
        "lineHeight": "1.2",
        "fontWeight": "500",
        "margin": "0 0 20px 0",
        "letterSpacing": "-2px",
        "color": "#000000",
    },
    "hero_desc": {
        "fontSize": "1.05rem",
        "lineHeight": "1.8",
        "color": "#666666",
        "maxWidth": "700px",
        "marginBottom": "30px",
    },
    "deco_square": {
        "width": "16px",
        "height": "16px",
        "background": "#FF4500",
        "marginBottom": "40px",
    },
    "metrics_row": {
        "display": "flex",
        "gap": "20px",
        "flexWrap": "wrap",
        "marginBottom": "40px",
    },
    "metric_card": {
        "border": "1px solid #E5E5E5",
        "padding": "20px 30px",
        "minWidth": "140px",
        "flex": "1",
    },
    "metric_value": {
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "1.8rem",
        "fontWeight": "520",
        "marginBottom": "5px",
        "color": "#000000",
    },
    "metric_label": {
        "fontSize": "0.75rem",
        "color": "#999999",
        "fontFamily": "'JetBrains Mono', monospace",
        "textTransform": "uppercase",
        "letterSpacing": "1px",
    },
    "tabs_wrap": {
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "0 40px",
        "borderTop": "1px solid #E5E5E5",
        "paddingTop": "30px",
    },
    "content": {
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "30px 40px 60px",
    },
    "card": {
        "border": "1px solid #E5E5E5",
        "padding": "8px",
        "marginBottom": "20px",
        "background": "#FFFFFF",
    },
    "card_inner": {
        "padding": "16px",
    },
    "stat_card": {
        "border": "1px solid #E5E5E5",
        "padding": "20px",
        "background": "#FAFAFA",
    },
    "section_label": {
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "0.75rem",
        "color": "#999999",
        "marginBottom": "8px",
        "display": "flex",
        "alignItems": "center",
        "gap": "8px",
    },
    "grid_2": {
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "20px",
    },
    "grid_3": {
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr 1fr",
        "gap": "16px",
    },
    # ── Insight button & panel ──
    "insight_btn": {
        "background": "#FFFFFF",
        "border": "1px solid #E5E5E5",
        "color": "#999999",
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "0.7rem",
        "fontWeight": "700",
        "padding": "4px 10px",
        "cursor": "pointer",
        "letterSpacing": "1px",
        "transition": "all 0.2s",
        "display": "inline-flex",
        "alignItems": "center",
        "gap": "6px",
    },
    "insight_panel": {
        "background": "#FAFAFA",
        "border": "1px solid #E5E5E5",
        "borderLeft": "3px solid #FF4500",
        "padding": "16px 20px",
        "marginTop": "12px",
        "fontSize": "0.9rem",
        "lineHeight": "1.7",
        "color": "#333333",
    },
    "insight_title": {
        "fontFamily": "'JetBrains Mono', monospace",
        "fontWeight": "700",
        "fontSize": "0.85rem",
        "color": "#000000",
        "marginBottom": "8px",
    },
}

TAB_STYLE = {
    "fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem",
    "fontWeight": "600", "border": "1px solid #E5E5E5",
    "borderBottom": "none", "background": "#FAFAFA", "color": "#666",
    "padding": "10px 24px",
}
TAB_SELECTED = {
    "fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem",
    "fontWeight": "700", "border": "1px solid #E5E5E5",
    "borderBottom": "2px solid #FF4500", "background": "#FFFFFF",
    "color": "#000000", "padding": "10px 24px",
}


def metric_card(value, label):
    return html.Div(style=S["metric_card"], children=[
        html.Div(str(value), style=S["metric_value"]),
        html.Div(label, style=S["metric_label"]),
    ])


def section_label(text):
    return html.Div(style=S["section_label"], children=[
        html.Span("◇", style={"fontSize": "1.2rem", "lineHeight": "1"}),
        text,
    ])


# ── Unique counter for card IDs ──
_card_counter = [0]


def _wrap_card(children, label=None, insight_key=None):
    """Wrap content in a MiroFish card with optional insight toggle button."""
    _card_counter[0] += 1
    card_id = f"card-{_card_counter[0]}"

    header_items = []
    if label:
        header_items.append(html.Span([
            html.Span("■ ", style={"color": "#FF4500", "fontSize": "0.8rem"}),
            label,
        ], style=S["section_label"]))

    # Add insight button if key exists
    if insight_key and insight_key in INSIGHTS:
        header_items.append(
            html.Button(
                ["? ", "Explain this chart"],
                id={"type": "insight-btn", "index": card_id},
                style=S["insight_btn"],
                n_clicks=0,
            )
        )

    header = html.Div(
        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
               "marginBottom": "8px"},
        children=header_items,
    ) if header_items else None

    # Insight panel (hidden by default)
    insight_panel = None
    if insight_key and insight_key in INSIGHTS:
        ins = INSIGHTS[insight_key]
        insight_panel = html.Div(
            id={"type": "insight-panel", "index": card_id},
            style={**S["insight_panel"], "display": "none"},
            children=[
                html.Div(ins["title"], style=S["insight_title"]),
                dcc.Markdown(ins["text"], style={"margin": "0", "fontSize": "0.9rem",
                             "lineHeight": "1.7", "color": "#333"}),
            ],
        )

    inner_children = []
    if header:
        inner_children.append(header)
    if insight_panel:
        inner_children.append(insight_panel)
    if isinstance(children, list):
        inner_children.extend(children)
    else:
        inner_children.append(children)

    return html.Div(style=S["card"], children=[
        html.Div(style=S["card_inner"], children=inner_children),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div(style=S["page"], children=[
    # ── Black Navbar ──
    html.Nav(style=S["navbar"], children=[
        html.Div("NOBEL PRIZE ANALYTICS", style=S["brand"]),
        html.Div(style=S["nav_meta"], children=[
            html.Span(f"{STATS['total_prizes']} prizes"),
            html.Span("·"),
            html.Span(f"{STATS['unique_laureates']} laureates"),
            html.Span("·"),
            html.Span(f"{STATS['year_range'][0]}–{STATS['year_range'][1]}"),
        ]),
    ]),

    # ── Hero ──
    html.Div(style=S["hero"], children=[
        html.Div(style=S["tag_row"], children=[
            html.Span("Interactive Data Science Platform", style=S["orange_tag"]),

        ]),
        html.H1("Nobel Prize", style=S["main_title"]),
        html.H1("Analytics Engine", className="gradient-text", style={
            **S["main_title"],
            "marginTop": "-15px",
        }),
        html.P([
            "Statistical analysis, machine learning & deep learning on ",
            html.Span(f"{STATS['unique_laureates']} Nobel laureates", style={
                "color": "#FF4500", "fontWeight": "700",
                "fontFamily": "'JetBrains Mono', monospace",
            }),
            f" across {len(CATEGORIES)} categories spanning "
            f"{STATS['year_range'][1] - STATS['year_range'][0] + 1} years of history.",
        ], style=S["hero_desc"]),

        # ── Central Research Question ──
        html.Div(style={
            "border": "2px solid #FF4500",
            "padding": "24px 32px",
            "marginBottom": "30px",
            "maxWidth": "800px",
            "position": "relative",
        }, children=[
            html.Span("RESEARCH QUESTION", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "0.65rem", "fontWeight": "700",
                "color": "#FFFFFF", "letterSpacing": "2px",
                "background": "#FF4500", "padding": "3px 10px",
                "position": "absolute", "top": "-10px", "left": "20px",
            }),
            html.P(
                "\"What shapes a Nobel laureate? How do geography, gender, age, and field "
                "interact across 125 years — and can data science predict the patterns of greatness?\"",
                style={
                    "fontSize": "1.15rem", "fontStyle": "italic",
                    "lineHeight": "1.8", "color": "#333", "margin": "0",
                    "fontWeight": "500",
                },
            ),
        ]),

        html.Div(style=S["deco_square"]),
        html.Div(style=S["metrics_row"], children=[
            metric_card(STATS["total_prizes"], "Total Prizes"),
            metric_card(STATS["unique_laureates"], "Laureates"),
            metric_card(STATS["countries"], "Countries"),
            metric_card(f"{STATS['avg_age']}y", "Avg Age"),
            metric_card(f"{STATS['female_pct']}%", "Female"),
            metric_card(len(CATEGORIES), "Categories"),
        ]),
    ]),

    # ── Tabs ──
    html.Div(style=S["tabs_wrap"], children=[
        section_label("Dashboard Modules"),
        dcc.Tabs(id="main-tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview", style=TAB_STYLE, selected_style=TAB_SELECTED),

            dcc.Tab(label="Demographics", value="demographics", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Trends & Stats", value="trends", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Machine Learning", value="ml", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Deep Dive", value="deep", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Gallery", value="gallery", style=TAB_STYLE, selected_style=TAB_SELECTED),
        ]),
    ]),

    # ── Content ──
    html.Div(id="tab-content", style=S["content"]),

    # ── Footer — Data Source ──
    html.Footer(style={
        "borderTop": "1px solid #E5E5E5",
        "padding": "30px 40px",
        "maxWidth": "1400px",
        "margin": "0 auto",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "flexWrap": "wrap",
        "gap": "12px",
    }, children=[
        html.Div([
            html.Span("DATA SOURCE", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "0.7rem", "fontWeight": "700",
                "color": "#999", "letterSpacing": "1px",
                "marginRight": "12px",
            }),
            html.A(
                "Nobel Prize API v2.1",
                href="https://api.nobelprize.org/2.1/laureates",
                target="_blank",
                style={
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize": "0.8rem", "color": "#FF4500",
                    "textDecoration": "none", "fontWeight": "600",
                },
            ),
            html.Span(
                f"  ·  {STATS['total_prizes']} records fetched live from nobelprize.org",
                style={
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize": "0.75rem", "color": "#999",
                },
            ),
        ]),
        html.Div([
            html.Span("© 2026 Nobel Prize Analytics", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "0.7rem", "color": "#BBBBBB",
            }),
        ]),
    ]),
])


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Narrative chapter intros
# ══════════════════════════════════════════════════════════════════════════════

def _chapter_intro(chapter_num, question, description):
    """Render a narrative chapter header that poses a question."""
    return html.Div(style={
        "marginBottom": "30px",
        "paddingBottom": "20px",
        "borderBottom": "1px solid #E5E5E5",
    }, children=[
        html.Span(f"CHAPTER {chapter_num}", style={
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize": "0.65rem", "fontWeight": "700",
            "color": "#FFFFFF", "letterSpacing": "2px",
            "background": "#000000", "padding": "3px 10px",
        }),
        html.H2(question, style={
            "fontSize": "1.6rem", "fontWeight": "500",
            "marginTop": "16px", "marginBottom": "8px",
            "letterSpacing": "-0.5px", "color": "#000",
        }),
        html.P(description, style={
            "fontSize": "0.95rem", "color": "#666",
            "lineHeight": "1.7", "maxWidth": "700px", "margin": "0",
        }),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# TAB CONTENT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _overview_tab():
    _card_counter[0] = 0  # Reset for consistent IDs per tab
    return html.Div([
        _chapter_intro(
            "01",
            "Where in the world does greatness concentrate?",
            "We begin with the big picture — mapping every Nobel Prize across countries, "
            "categories, and time. The patterns of geographic dominance become immediately clear.",
        ),
        _wrap_card(dcc.Graph(figure=viz.fig_globe_3d(DF), config={"scrollZoom": True}), "01 / 3D Globe — World Distribution", "globe_3d"),
        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_timeline_cumulative(DF)), "02 / Timeline", "timeline"),
            _wrap_card(dcc.Graph(figure=viz.fig_prizes_by_category(DF)), "03 / Categories", "categories"),
        ]),
        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_top_countries(DF)), "04 / Top Countries", "top_countries"),
            _wrap_card(dcc.Graph(figure=viz.fig_sunburst(DF)), "05 / Hierarchy", "sunburst"),
        ]),
    ])


def _demographics_tab():
    _card_counter[0] = 100
    return html.Div([
        _chapter_intro(
            "03",
            "Who wins — and who is left out?",
            "Nobel Prizes reflect not just individual brilliance but systemic access to education, "
            "funding, and opportunity. Here we examine how gender, age, and lifespan reveal "
            "the evolving — but still unequal — face of scientific recognition.",
        ),
        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_gender_decade(DF)), "01 / Gender × Decade", "gender_decade"),
            _wrap_card(dcc.Graph(figure=viz.fig_gender_pct_trend(DF)), "02 / Female % Trend", "gender_trend"),
        ]),
        _wrap_card(dcc.Graph(figure=viz.fig_age_scatter(DF)), "03 / Age Over Time", "age_scatter"),
        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_age_violin(DF)), "04 / Age Distribution", "age_violin"),
            _wrap_card(dcc.Graph(figure=viz.fig_age_histogram(DF)), "05 / Age Histogram", "age_histogram"),
        ]),
        _wrap_card(dcc.Graph(figure=viz.fig_lifespan_box(DF)), "06 / Lifespan", "lifespan"),
    ])


def _trends_tab():
    _card_counter[0] = 200
    return html.Div([
        _chapter_intro(
            "04",
            "Are the patterns real, or just coincidence?",
            "Intuition can be misleading. Statistical hypothesis testing lets us separate genuine "
            "trends from noise. We test whether gender imbalance and age differences are "
            "statistically significant, and track how prize patterns have shifted over decades.",
        ),
        _wrap_card([
            html.Div(style=S["grid_3"], children=[
                html.Div(style=S["stat_card"], children=[
                    html.Div("Chi-Squared: Gender × Category", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "0.75rem", "color": "#999", "marginBottom": "12px"}),
                    html.Div(f"χ² = {chi['chi2_statistic']}", style={
                        "fontSize": "1.5rem", "fontWeight": "520",
                        "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div(f"p = {chi['p_value']:.2e}", style={"color": "#666", "fontSize": "0.85rem"}),
                    html.Div(
                        "● Significant" if chi["significant"] else "○ Not significant",
                        style={"color": "#FF4500" if chi["significant"] else "#999",
                               "fontFamily": "'JetBrains Mono', monospace",
                               "fontSize": "0.8rem", "marginTop": "8px"}),
                ]),
                html.Div(style=S["stat_card"], children=[
                    html.Div("Mann-Whitney: Physics vs Literature", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "0.75rem", "color": "#999", "marginBottom": "12px"}),
                    html.Div(f"U = {mw['U_statistic']}", style={
                        "fontSize": "1.5rem", "fontWeight": "520",
                        "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div(f"p = {mw['p_value']:.2e}", style={"color": "#666", "fontSize": "0.85rem"}),
                    html.Div(
                        "● Significant" if mw["significant"] else "○ Not significant",
                        style={"color": "#FF4500" if mw["significant"] else "#999",
                               "fontFamily": "'JetBrains Mono', monospace",
                               "fontSize": "0.8rem", "marginTop": "8px"}),
                    html.Div(f"Medians: {mw['median_1']} vs {mw['median_2']}y", style={
                        "color": "#999", "fontSize": "0.75rem",
                        "fontFamily": "'JetBrains Mono', monospace", "marginTop": "4px"}),
                ]),
                html.Div(style=S["stat_card"], children=[
                    html.Div("Age Distribution", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "0.75rem", "color": "#999", "marginBottom": "12px"}),
                    html.Div(f"μ={age_stats['mean']}  σ={age_stats['std']}", style={
                        "fontSize": "1.2rem", "fontWeight": "520",
                        "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div(f"Skew={age_stats['skewness']}  Kurt={age_stats['kurtosis']}", style={
                        "color": "#666", "fontSize": "0.85rem",
                        "fontFamily": "'JetBrains Mono', monospace"}),
                    html.Div(
                        f"{'● Normal' if age_stats['is_normal'] else '○ Non-normal'} "
                        f"(Shapiro p={age_stats['shapiro_p']:.2e})",
                        style={"color": "#FF4500" if not age_stats["is_normal"] else "#999",
                               "fontFamily": "'JetBrains Mono', monospace",
                               "fontSize": "0.8rem", "marginTop": "8px"}),
                ]),
            ]),
        ], "Hypothesis Tests", "hypothesis_tests"),

        _wrap_card(dcc.Graph(figure=viz.fig_regression_scatter(reg, DF)),
                   f"OLS Regression  /  slope={reg['slope']:.4f}  R²={reg['r_squared']:.4f}",
                   "regression"),

        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_category_decade_heatmap(DF)), "Heatmap", "heatmap"),
            _wrap_card(dcc.Graph(figure=viz.fig_correlation_heatmap(ANALYSIS["correlations"])),
                       "Correlations", "correlations"),
        ]),
        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_prize_amount_trend(DF)), "Prize Amount Trend", "prize_amount"),
            _wrap_card(dcc.Graph(figure=viz.fig_category_trends_line(DF)), "Category Trends", "category_trends"),
        ]),
    ])


def _ml_tab():
    _card_counter[0] = 300
    topic_figs = []
    for i in range(min(4, len(tp["topics"]))):
        topic_figs.append(
            _wrap_card(dcc.Graph(figure=viz.fig_topic_words(tp, i)))
        )

    return html.Div([
        _chapter_intro(
            "05",
            "Can machines learn the formula behind a Nobel Prize?",
            "We unleash machine learning and deep learning on the data — clustering laureates "
            "into hidden groups, predicting prize categories from demographics, mining the language "
            "of prize motivations, and forecasting the future with neural networks.",
        ),
        _wrap_card([
            html.Div(
                f"Grouped {len(cl['data'])} laureates into {cl['n_clusters']} clusters. "
                f"Silhouette = {cl['best_silhouette']:.4f}",
                style={"color": "#666", "marginBottom": "12px",
                       "fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem"}),
        ], "01 / K-Means Clustering", "kmeans"),
        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_clusters(cl)), insight_key="kmeans"),
            _wrap_card(dcc.Graph(figure=viz.fig_silhouette_scores(cl)), insight_key="silhouette"),
        ]),

        _wrap_card([
            html.Div(
                f"5-fold CV accuracy: {rf['cv_accuracy_mean']:.1%} ± {rf['cv_accuracy_std']:.1%}  ·  "
                f"{rf['n_samples']} samples  ·  {len(rf['categories'])} categories",
                style={"color": "#666", "marginBottom": "12px",
                       "fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem"}),
            dcc.Graph(figure=viz.fig_feature_importances(rf)),
        ], "02 / Random Forest Classification", "random_forest"),

        _wrap_card([
            html.Div(
                f"{len(tp['topics'])} topics from {tp['n_documents']} motivations  ·  "
                f"{tp['explained_variance']}% variance explained",
                style={"color": "#666", "marginBottom": "12px",
                       "fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem"}),
            dcc.Graph(figure=viz.fig_topic_heatmap(tp)),
        ], "03 / NLP Topic Modeling", "nlp_topics"),
        html.Div(style=S["grid_2"], children=topic_figs[:4]),

        _wrap_card([
            html.Div(
                f"2-layer LSTM  ·  Final MSE: {fc['final_loss']:.6f}  ·  "
                f"Forecasting {len(fc['forecast_years'])} years ahead",
                style={"color": "#666", "marginBottom": "12px",
                       "fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem"}),
        ], "04 / LSTM Forecasting (PyTorch)", "lstm_forecast"),
        html.Div(style=S["grid_2"], children=[
            _wrap_card(dcc.Graph(figure=viz.fig_lstm_forecast(fc)), insight_key="lstm_forecast"),
            _wrap_card(dcc.Graph(figure=viz.fig_training_loss(fc)), insight_key="training_loss"),
        ]),
    ])


def _deep_tab():
    _card_counter[0] = 400
    return html.Div([
        _chapter_intro(
            "06",
            "What stories hide in the connections between countries and fields?",
            "We go deeper with advanced visualizations — tracing the flow of laureates across continents, "
            "breaking down the hierarchy of regions and categories, and spotlighting "
            "the rare few who achieved the extraordinary distinction of winning twice.",
        ),
        _wrap_card(dcc.Graph(figure=viz.fig_sankey(DF)), "01 / Continent → Category Flow", "sankey"),
        _wrap_card(dcc.Graph(figure=viz.fig_treemap(DF)), "02 / Treemap", "treemap"),
        _wrap_card(dcc.Graph(figure=viz.fig_multi_laureates(DF)), "03 / Multi-Prize Laureates", "multi_laureates"),
    ])






def _gallery_tab():
    _card_counter[0] = 600
    
    decades = [{"label": f"{d}s", "value": d} for d in sorted(DF["decade"].dropna().unique())]
    categories = [{"label": c, "value": c} for c in CATEGORIES]
    birth_countries = [{"label": c, "value": c} for c in sorted(DF["birth_country"].dropna().unique())]
    
    all_nats = set()
    if "nationality" in DF.columns:
        for n in DF["nationality"].dropna():
            for part in n.split(", "):
                all_nats.add(part)
    nationalities = [{"label": c, "value": c} for c in sorted(all_nats)]
    
    return html.Div([
        _chapter_intro(
            "07",
            "Who are the people behind the prizes?",
            "Explore the faces and stories of all 1018 Nobel laureates across 125 years of history. "
            "From brilliant physicists to inspiring peacemakers, browse their details and Wikipedia portraits."
        ),
        _wrap_card(
            html.Div([
                dcc.Store(id="gallery-page", data=1),
                
                # Filters
                html.Div(style={
                    "display": "flex", "gap": "16px", "marginBottom": "24px",
                    "flexWrap": "wrap", "background": "#FAFAFA", "padding": "16px",
                    "border": "1px solid #EAEAEA", "borderRadius": "8px"
                }, children=[
                    html.Div([
                        html.Span("Search Name or Work:", style={"display": "block", "fontSize": "0.75rem", "fontFamily": "'JetBrains Mono', monospace", "color": "#666", "marginBottom": "4px"}),
                        dcc.Input(id="gallery-search", type="text", placeholder="e.g. Einstein", style={"padding": "8px", "width": "200px", "border": "1px solid #CCC", "borderRadius": "4px", "fontFamily": "inherit"})
                    ]),
                    html.Div([
                        html.Span("Category:", style={"display": "block", "fontSize": "0.75rem", "fontFamily": "'JetBrains Mono', monospace", "color": "#666", "marginBottom": "4px"}),
                        dcc.Dropdown(id="gallery-category", options=[{"label": "All", "value": "All"}] + categories, value="All", clearable=False, style={"width": "180px"})
                    ]),
                    html.Div([
                        html.Span("Decade:", style={"display": "block", "fontSize": "0.75rem", "fontFamily": "'JetBrains Mono', monospace", "color": "#666", "marginBottom": "4px"}),
                        dcc.Dropdown(id="gallery-decade", options=[{"label": "All", "value": "All"}] + decades, value="All", clearable=False, style={"width": "120px"})
                    ]),
                    html.Div([
                        html.Span("Birth Country:", style={"display": "block", "fontSize": "0.75rem", "fontFamily": "'JetBrains Mono', monospace", "color": "#666", "marginBottom": "4px"}),
                        dcc.Dropdown(id="gallery-birth-country", options=[{"label": "All", "value": "All"}] + birth_countries, value="All", clearable=False, style={"width": "180px"})
                    ]),
                    html.Div([
                        html.Span("Nationality:", style={"display": "block", "fontSize": "0.75rem", "fontFamily": "'JetBrains Mono', monospace", "color": "#666", "marginBottom": "4px"}),
                        dcc.Dropdown(id="gallery-nationality", options=[{"label": "All", "value": "All"}] + nationalities, value="All", clearable=False, style={"width": "180px"})
                    ]),
                ]),
                
                # Pagination Controls
                html.Div(style={
                    "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                    "marginBottom": "24px", "paddingBottom": "16px",
                    "borderBottom": "1px solid #EAEAEA"
                }, children=[
                    html.Button("← Previous", id="btn-prev-page", n_clicks=0, style={
                        "padding": "8px 16px", "background": "#FFFFFF", "border": "1px solid #CCC",
                        "cursor": "pointer", "fontFamily": "'JetBrains Mono', monospace",
                        "borderRadius": "4px", "fontWeight": "600"
                    }),
                    html.Span(id="gallery-page-info", style={
                        "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "600",
                        "color": "#333", "fontSize": "0.95rem"
                    }),
                    html.Button("Next →", id="btn-next-page", n_clicks=0, style={
                        "padding": "8px 16px", "background": "#111", "color": "#FFF", "border": "none",
                        "cursor": "pointer", "fontFamily": "'JetBrains Mono', monospace",
                        "borderRadius": "4px", "fontWeight": "600"
                    }),
                ]),
                
                # The Grid container
                dcc.Loading(
                    html.Div(id="gallery-grid", style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fill, minmax(280px, 1fr))",
                        "gap": "24px",
                    })
                )
            ]),
            "01 / Laureates Archive",
            "gallery_archive"
        )
    ])


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    tab_map = {
        "overview": _overview_tab,

        "demographics": _demographics_tab,
        "trends": _trends_tab,
        "ml": _ml_tab,
        "deep": _deep_tab,
        "gallery": _gallery_tab,
    }
    builder = tab_map.get(tab)
    if builder is None:
        return html.Div("Select a tab")

    tab_content = builder()

    # ── Conclusion panel ──
    conclusion = html.Div(style={
        "background": "#000",
        "padding": "40px 40px",
        "marginTop": "40px",
        "position": "relative",
    }, children=[
        html.Span("CONCLUSION", style={
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize": "0.65rem", "fontWeight": "700",
            "color": "#000", "letterSpacing": "2px",
            "background": "#FF4500", "padding": "3px 10px",
            "display": "inline-block", "marginBottom": "16px",
        }),
        html.H3(
            "So, what shapes a Nobel laureate?",
            style={
                "color": "#FFFFFF", "fontSize": "1.5rem",
                "fontWeight": "500", "marginBottom": "16px",
                "letterSpacing": "-0.5px",
            },
        ),
        html.P([
            f"Across {STATS['year_range'][1] - STATS['year_range'][0] + 1} years and "
            f"{STATS['total_prizes']} prizes, the data reveals a clear story: ",
            html.Span("geography matters", style={"color": "#FF4500", "fontWeight": "700"}),
            f" — {STATS['countries']} countries have produced laureates, but a handful dominate. ",
            html.Span("Gender gaps persist", style={"color": "#FF4500", "fontWeight": "700"}),
            f" — only {STATS['female_pct']}% of laureates are women. ",
            html.Span("Age tells a pattern", style={"color": "#FF4500", "fontWeight": "700"}),
            f" — the average laureate is {STATS['avg_age']} years old. ",
            "And machine learning confirms that these patterns are not random — "
            "they reflect systemic forces in education, funding, and global opportunity.",
        ], style={
            "color": "#CCCCCC", "fontSize": "0.95rem",
            "lineHeight": "1.8", "maxWidth": "800px", "margin": "0",
        }),
        html.P(
            "The Nobel Prize is not just a celebration of individual genius — "
            "it is a mirror of the world's evolving scientific ecosystem.",
            style={
                "color": "#FFFFFF", "fontSize": "1.05rem",
                "fontWeight": "500", "fontStyle": "italic",
                "marginTop": "20px", "maxWidth": "700px",
            },
        ),
    ])

    return html.Div([tab_content, conclusion])





@callback(
    Output({"type": "insight-panel", "index": MATCH}, "style"),
    Input({"type": "insight-btn", "index": MATCH}, "n_clicks"),
    State({"type": "insight-panel", "index": MATCH}, "style"),
    prevent_initial_call=True,
)
def toggle_insight(n_clicks, current_style):
    """Toggle visibility of insight panel on button click."""
    if current_style is None:
        current_style = {}
    is_hidden = current_style.get("display") == "none"
    new_style = {**S["insight_panel"], "display": "block" if is_hidden else "none"}
    return new_style


@callback(
    Output("gallery-grid", "children"),
    Output("gallery-page-info", "children"),
    Output("gallery-page", "data"),
    Input("main-tabs", "value"),
    Input("btn-prev-page", "n_clicks"),
    Input("btn-next-page", "n_clicks"),
    Input("gallery-search", "value"),
    Input("gallery-category", "value"),
    Input("gallery-decade", "value"),
    Input("gallery-birth-country", "value"),
    Input("gallery-nationality", "value"),
    State("gallery-page", "data"),
    prevent_initial_call=False,
)
def update_gallery(tab, prev_clicks, next_clicks, search, category, decade, birth_country, nationality, current_page):
    if tab != "gallery":
        return dash.no_update, dash.no_update, dash.no_update
        
    from dash import ctx
    triggered = ctx.triggered_id
    
    # Filter the DataFrame based on search, category, decade, countries
    filtered_df = DF.copy()
    if search:
        search_lower = search.lower()
        filtered_df = filtered_df[filtered_df["name"].str.lower().str.contains(search_lower) | 
                                  filtered_df["motivation"].str.lower().str.contains(search_lower)]
    if category and category != "All":
        filtered_df = filtered_df[filtered_df["category"] == category]
    if decade and decade != "All":
        filtered_df = filtered_df[filtered_df["decade"] == int(decade)]
    if birth_country and birth_country != "All":
        filtered_df = filtered_df[filtered_df["birth_country"] == birth_country]
    if nationality and nationality != "All":
        filtered_df = filtered_df[filtered_df["nationality"].str.contains(nationality, na=False, regex=False)]
        
    ITEMS_PER_PAGE = 40
    total_items = len(filtered_df)
    max_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    
    # Check trigger logic to reset page or advance it
    if current_page is None:
        current_page = 1
        
    if triggered in ["gallery-search", "gallery-category", "gallery-decade", "gallery-birth-country", "gallery-nationality"]:
        current_page = 1
    elif triggered == "btn-next-page" and current_page < max_pages:
        current_page += 1
    elif triggered == "btn-prev-page" and current_page > 1:
        current_page -= 1
        
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    cards = []
    if len(page_df) == 0:
        cards = [html.Div("No laureates match your search filters.", style={"gridColumn": "1 / -1", "padding": "40px", "textAlign": "center", "color": "#666", "fontSize": "1.2rem"})]
    else:
        for _, row in page_df.iterrows():
            img_url = row.get("img_url", "")
            
            if not img_url:
                img_element = html.Div(
                    "?", 
                    style={
                        "width": "100%", "height": "240px", "background": "#F5F5F5",
                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                        "fontSize": "4rem", "color": "#DDD", "borderBottom": "1px solid #EAEAEA",
                        "fontFamily": "Inter, sans-serif", "fontWeight": "800"
                    }
                )
            else:
                img_element = html.Div(style={
                    "width": "100%", "height": "240px", 
                    "backgroundImage": f"url('{img_url}')",
                    "backgroundSize": "cover",
                    "backgroundPosition": "center",
                    "borderBottom": "1px solid #EAEAEA"
                })
                
            color = viz.COLORS.get(row["category"], "#333")
                
            card = html.Div(style={
                "border": "1px solid #EAEAEA", "borderRadius": "8px",
                "overflow": "hidden", "background": "#FFFFFF",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.03)",
                "display": "flex", "flexDirection": "column",
                "transition": "transform 0.2s ease",
            }, children=[
                img_element,
                html.Div(style={"padding": "20px"}, children=[
                    html.Span(row["category"].upper(), style={
                        "fontSize": "0.65rem", "fontWeight": "800", "color": color,
                        "letterSpacing": "1.5px", "marginBottom": "8px", "display": "block"
                    }),
                    html.H4(row["name"], style={
                        "margin": "0 0 8px 0", "fontSize": "1.15rem", "color": "#000",
                        "fontWeight": "600"
                    }),
                    html.P(f"Awarded: {row['award_year']} · {row.get('nationality') if pd.notna(row.get('nationality')) else (row['birth_country'] or 'Unknown')}", style={
                        "fontSize": "0.75rem", "color": "#666", "margin": "0 0 16px 0",
                        "fontFamily": "'JetBrains Mono', monospace"
                    }),
                    html.P(row["motivation"], style={
                        "fontSize": "0.9rem", "lineHeight": "1.6", "color": "#444",
                        "margin": "0", "display": "-webkit-box", "-webkit-line-clamp": "4",
                        "-webkit-box-orient": "vertical", "overflow": "hidden"
                    })
                ])
            ])
            cards.append(card)
        
    page_info = f"Page {current_page} of {max_pages} · {total_items} results"
    return cards, page_info, current_page


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NOBEL PRIZE ANALYTICS")
    print(f"   {STATS['total_prizes']} prizes | {STATS['unique_laureates']} laureates")
    print(f"   {STATS['year_range'][0]}–{STATS['year_range'][1]}")
    print("=" * 60)
    print("\n→ Open http://localhost:8050 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
