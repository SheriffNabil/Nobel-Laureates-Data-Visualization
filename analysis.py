"""
analysis.py — Statistical Analysis of Nobel Laureate Data
Includes descriptive stats, hypothesis tests, regression, and time-series analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def descriptive_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics per prize category."""
    persons = df[~df["is_org"]].copy()
    return (
        persons.groupby("category")
        .agg(
            count=("id", "count"),
            unique_laureates=("id", "nunique"),
            avg_age=("age_at_award", "mean"),
            median_age=("age_at_award", "median"),
            std_age=("age_at_award", "std"),
            min_age=("age_at_award", "min"),
            max_age=("age_at_award", "max"),
            pct_female=("gender", lambda x: round((x == "female").mean() * 100, 1)),
            avg_lifespan=("lifespan", "mean"),
        )
        .round(1)
        .sort_values("count", ascending=False)
    )


def gender_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Gender distribution over decades."""
    persons = df[~df["is_org"]].copy()
    ct = pd.crosstab(persons["decade"], persons["gender"])
    if "female" not in ct.columns:
        ct["female"] = 0
    ct["total"] = ct.sum(axis=1)
    ct["female_pct"] = round(ct.get("female", 0) / ct["total"] * 100, 1)
    return ct


def country_rankings(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Rank countries by total Nobel prizes."""
    return (
        df.groupby("birth_country")
        .agg(count=("id", "count"), categories=("category", "nunique"))
        .sort_values("count", ascending=False)
        .head(top_n)
    )


def chi_squared_gender_category(df: pd.DataFrame) -> dict:
    """
    Chi-squared test for independence between gender and category.
    H0: Gender distribution is independent of prize category.
    """
    persons = df[~df["is_org"] & df["gender"].isin(["male", "female"])].copy()
    ct = pd.crosstab(persons["gender"], persons["category"])
    chi2, p_value, dof, expected = stats.chi2_contingency(ct)
    return {
        "test": "Chi-squared test for Gender × Category independence",
        "chi2_statistic": round(chi2, 2),
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "significant": p_value < 0.05,
        "interpretation": (
            "Gender distribution is significantly different across categories"
            if p_value < 0.05
            else "No significant difference in gender distribution across categories"
        ),
    }


def mann_whitney_age(
    df: pd.DataFrame, cat1: str = "Physics", cat2: str = "Literature"
) -> dict:
    """
    Mann-Whitney U test comparing age at award between two categories.
    H0: The distributions of ages are the same.
    """
    persons = df[~df["is_org"]].copy()
    a1 = persons[persons["category"] == cat1]["age_at_award"].dropna()
    a2 = persons[persons["category"] == cat2]["age_at_award"].dropna()
    stat, p_value = stats.mannwhitneyu(a1, a2, alternative="two-sided")
    return {
        "test": f"Mann-Whitney U test: {cat1} vs {cat2} age at award",
        "U_statistic": round(stat, 2),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "median_1": round(a1.median(), 1),
        "median_2": round(a2.median(), 1),
        "interpretation": (
            f"{cat1} and {cat2} laureates have significantly different award ages"
            if p_value < 0.05
            else f"No significant age difference between {cat1} and {cat2} laureates"
        ),
    }


def age_trend_regression(df: pd.DataFrame) -> dict:
    """
    OLS linear regression: age_at_award ~ award_year.
    Tests whether laureates are getting older over time.
    """
    persons = df[~df["is_org"]].dropna(subset=["age_at_award", "award_year"]).copy()
    X = persons["award_year"].values.reshape(-1, 1)
    y = persons["age_at_award"].values

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Statistical significance of slope
    n = len(y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    sx = np.sqrt(np.sum((X.ravel() - X.mean()) ** 2))
    se_slope = se / sx
    t_stat = slope / se_slope
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 2)

    return {
        "test": "OLS Regression: Age at Award ~ Year",
        "slope": round(slope, 4),
        "intercept": round(intercept, 2),
        "r_squared": round(r2, 4),
        "t_statistic": round(t_stat, 2),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "interpretation": (
            f"Age at award increases by ~{slope:.2f} years per year (significant trend)"
            if p_value < 0.05
            else "No significant trend in age at award over time"
        ),
        "predictions": {
            "year": persons["award_year"].values.tolist(),
            "actual": y.tolist(),
            "predicted": y_pred.tolist(),
        },
    }


def lifespan_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Average lifespan by category for deceased laureates."""
    deceased = df[~df["is_org"] & df["lifespan"].notna()].copy()
    return (
        deceased.groupby("category")
        .agg(
            avg_lifespan=("lifespan", "mean"),
            median_lifespan=("lifespan", "median"),
            count=("id", "count"),
        )
        .round(1)
        .sort_values("avg_lifespan", ascending=False)
    )


def prizes_per_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Count prizes per decade per category."""
    return (
        df.groupby(["decade", "category"])
        .agg(count=("id", "count"))
        .reset_index()
        .pivot(index="decade", columns="category", values="count")
        .fillna(0)
        .astype(int)
    )


def age_distribution_stats(df: pd.DataFrame) -> dict:
    """Compute age distribution statistics and normality test."""
    persons = df[~df["is_org"]].copy()
    ages = persons["age_at_award"].dropna()

    # Shapiro-Wilk normality test (on a sample if > 5000)
    sample = ages.sample(min(len(ages), 5000), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample)

    # Skewness and kurtosis
    return {
        "mean": round(ages.mean(), 1),
        "median": round(ages.median(), 1),
        "std": round(ages.std(), 1),
        "skewness": round(ages.skew(), 3),
        "kurtosis": round(ages.kurtosis(), 3),
        "shapiro_stat": round(shapiro_stat, 4),
        "shapiro_p": shapiro_p,
        "is_normal": shapiro_p > 0.05,
        "youngest": int(ages.min()),
        "oldest": int(ages.max()),
    }


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix for numerical features."""
    persons = df[~df["is_org"]].copy()
    numeric_cols = ["award_year", "age_at_award", "prize_amount", "prize_amount_adjusted", "lifespan"]
    available = [c for c in numeric_cols if c in persons.columns]
    return persons[available].corr().round(3)


def run_analysis(df: pd.DataFrame) -> dict:
    """Run all statistical analyses and return results dict."""
    results = {
        "descriptive": descriptive_by_category(df),
        "gender_decades": gender_by_decade(df),
        "country_rankings": country_rankings(df),
        "chi_squared": chi_squared_gender_category(df),
        "mann_whitney": mann_whitney_age(df),
        "age_regression": age_trend_regression(df),
        "lifespan": lifespan_by_category(df),
        "prizes_decade": prizes_per_decade(df),
        "age_distribution": age_distribution_stats(df),
        "correlations": correlation_analysis(df),
    }
    return results


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    results = run_analysis(df)

    print("\n" + "=" * 60)
    print("📈 STATISTICAL ANALYSIS RESULTS")
    print("=" * 60)

    print("\n--- Chi-Squared Test ---")
    for k, v in results["chi_squared"].items():
        print(f"  {k}: {v}")

    print("\n--- Mann-Whitney U Test ---")
    for k, v in results["mann_whitney"].items():
        print(f"  {k}: {v}")

    print("\n--- Age Trend Regression ---")
    reg = results["age_regression"]
    for k, v in reg.items():
        if k != "predictions":
            print(f"  {k}: {v}")

    print("\n--- Age Distribution ---")
    for k, v in results["age_distribution"].items():
        print(f"  {k}: {v}")
