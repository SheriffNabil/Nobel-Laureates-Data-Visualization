# 🏅 Nobel Prize Analytics Engine

> An interactive data analytics dashboard exploring **125 years of Nobel Prize history** (1901–2025) with 3D globe visualizations, machine learning models, and deep statistical analysis — powered by live data from the Nobel Prize API.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-Plotly-000000?logo=plotly&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 📊 Six Interactive Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | 3D interactive globe, cumulative timeline, category breakdown, top countries, and continental hierarchy |
| **Demographics** | Gender distribution trends, age-at-award analysis (violin, scatter, histogram), lifespan statistics |
| **Trends & Stats** | Category × decade heatmap, prize amount trends (inflation-adjusted), Sankey flows, treemaps, category trend lines |
| **Machine Learning** | K-Means clustering, Random Forest classification, NLP topic modeling, LSTM time-series forecasting |
| **Deep Dive** | Correlation heatmap, OLS regression analysis, multi-prize laureates, and interactive exploration tools |
| **Gallery** | Searchable laureate gallery with Wikipedia portrait images and biographical details |

### 🌍 3D Globe Visualization
- Interactive orthographic globe with **real country coordinates**
- Bubble markers sized by prize count with hover details
- Drag-to-rotate, click-to-explore interaction

### 🤖 Machine Learning Pipeline
- **K-Means Clustering** — discovers latent patterns in laureate demographics
- **Random Forest Classifier** — predicts prize category from features (with feature importance analysis)
- **NLP Topic Modeling** — extracts themes from prize motivations using TF-IDF + NMF
- **LSTM Neural Network** — forecasts future prize counts using PyTorch

### 🎨 Design
- **MiroFish-inspired white theme** — clean monochrome with orange (#FF4500) accents
- **JetBrains Mono** typography for data elements
- Fully responsive layout with cross-browser compatibility (Chrome, Safari, Firefox)
- Every chart includes an AI-powered "Explain this chart" button

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/SheriffNabil/Nobel-Laureates-Data-Visualization.git
cd Nobel-Laureates-Data-Visualization

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The app will launch at **http://localhost:8050**

### Dependencies

| Package | Purpose |
|---------|---------|
| `dash` | Web framework and reactive UI |
| `plotly` | Interactive charts and 3D globe |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `scikit-learn` | K-Means, Random Forest, NMF, TF-IDF |
| `scipy` | Statistical analysis and regression |
| `torch` | LSTM neural network (PyTorch) |
| `nltk` | Natural language processing |
| `requests` | Nobel Prize API + Wikipedia data fetching |

---

## 🏗️ Architecture

```
Nobel-Laureates-Data-Visualization/
├── app.py              # Main Dash application — layout, styling, callbacks
├── data_loader.py      # Nobel API client, data enrichment, Wikipedia portraits
├── visualizations.py   # 20+ Plotly chart functions with MiroFish theme
├── ml_models.py        # ML/DL pipeline — clustering, classification, NLP, LSTM
├── analysis.py         # Statistical analysis — regression, correlations, tests
├── cleanup.py          # Data cleaning and preprocessing utilities
├── requirements.txt    # Python dependencies
└── README.md
```

### Data Pipeline

```
Nobel Prize API → data_loader.py → cleanup.py → analysis.py → ml_models.py
                                                      ↓              ↓
                                              visualizations.py ← app.py → Browser
```

1. **Fetch** — Live data from `api.nobelprize.org/2.1/laureates` (1000+ laureates)
2. **Enrich** — Wikipedia portraits, Wikidata nationalities, continent mapping
3. **Analyze** — Statistical tests, regression, correlation matrices
4. **Model** — K-Means, Random Forest, NLP topic extraction, LSTM forecasting
5. **Visualize** — 20+ interactive Plotly charts across 6 dashboard tabs

---

## 📈 Key Metrics

| Metric | Value |
|--------|-------|
| Total Prizes | 1,026 |
| Total Laureates | 1,018 |
| Year Range | 1901–2025 |
| Countries Represented | 60+ |
| Interactive Charts | 20+ |
| ML Models | 4 |

---

## 🛠️ Technical Details

### Live Data Ingestion
The application fetches real-time data from the official Nobel Prize API on startup — no static CSV files or stale datasets. Laureate portraits are batch-fetched from Wikipedia, and nationalities are enriched via Wikidata SPARQL queries.

### Machine Learning Models
All models train on app startup using the live data:
- **K-Means** (k=2–8, automatic silhouette score optimization)
- **Random Forest** (5-fold cross-validated, feature importance ranking)
- **NMF Topic Model** (TF-IDF vectorized motivations, variance explained tracking)
- **LSTM** (PyTorch, sliding window, MSE loss, 10-year forecast horizon)

### Visualization Theme
Every chart follows the **MiroFish white theme**: white background, black text, `#FF4500` orange accent color, `JetBrains Mono` for data labels, `Space Grotesk` for body text, and `#EAEAEA` grid lines.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Nobel Prize API](https://www.nobelprize.org/about/developer-zone-2/) — official data source
- [Wikipedia](https://www.wikipedia.org/) — laureate portraits
- [Wikidata](https://www.wikidata.org/) — nationality enrichment
- [Plotly Dash](https://dash.plotly.com/) — visualization framework
- [PyTorch](https://pytorch.org/) — LSTM forecasting

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/SheriffNabil">SheriffNabil</a>
</p>
