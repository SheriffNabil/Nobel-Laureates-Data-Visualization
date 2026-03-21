"""
data_loader.py — Nobel Laureates Data Pipeline
Fetches from Nobel Prize API v2.1, parses nested JSON into a clean DataFrame.
Falls back to local laureates.json if the API is unavailable.
"""

import json
import os
import re
import functools
from datetime import datetime

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.nobelprize.org/2.1/laureates?limit=1000"
LOCAL_FILE = os.path.join(os.path.dirname(__file__), "laureates.json")


def _fetch_from_api() -> dict:
    """Fetch laureate data from the Nobel Prize API with pagination."""
    headers = {
        "User-Agent": "NobelPrize-Analytics-Dashboard/1.0",
        "Accept": "application/json",
    }
    all_laureates = []
    url = API_URL
    while url:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        all_laureates.extend(data.get("laureates", []))
        # Handle pagination
        links = data.get("links", {})
        url = None
        if isinstance(links, list):
            for link in links:
                if link.get("rel") == "next":
                    url = link["href"]
                    break
        elif isinstance(links, dict):
            url = links.get("next")
    return {"laureates": all_laureates}


def _load_local() -> dict:
    """Load laureate data from local JSON file."""
    with open(LOCAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _parse_date(date_str: str):
    """Parse date string, handling partial dates like '1943-00-00'."""
    if not date_str or date_str == "None":
        return None
    date_str = date_str.replace("-00", "-01")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _compute_age(birth_date, award_year):
    """Compute age at the time of award."""
    if birth_date is None or award_year is None:
        return None
    try:
        return int(award_year) - birth_date.year
    except (ValueError, TypeError):
        return None


def _compute_lifespan(birth_date, death_date):
    """Compute lifespan in years."""
    if birth_date is None or death_date is None:
        return None
    try:
        return round((death_date - birth_date).days / 365.25, 1)
    except (ValueError, TypeError):
        return None


def load_data(use_api: bool = False) -> pd.DataFrame:
    """
    Load and parse Nobel laureate data into a flat DataFrame.
    
    Parameters
    ----------
    use_api : bool
        If True, attempt to fetch fresh data from the API first.
    
    Returns
    -------
    pd.DataFrame with one row per prize-laureate pair.
    """
    # Fetch data
    raw = None
    if use_api:
        try:
            raw = _fetch_from_api()
            print(f"✓ Fetched {len(raw['laureates'])} laureates from API")
        except Exception as e:
            print(f"⚠ API fetch failed ({e}), falling back to local file")

    if raw is None:
        raw = _load_local()
        print(f"✓ Loaded {len(raw['laureates'])} laureates from local file")

    # Parse into rows (one row per prize per laureate)
    rows = []
    for lau in raw["laureates"]:
        is_org = "orgName" in lau

        # Basic info
        if is_org:
            name = _safe_get(lau, "orgName", "en", default="Unknown Org")
            gender = "organization"
            birth_date_str = _safe_get(lau, "founded", "date", default=None)
            birth_city = _safe_get(lau, "founded", "place", "city", "en", default=None)
            birth_country = _safe_get(lau, "founded", "place", "country", "en", default=None)
            birth_continent = _safe_get(lau, "founded", "place", "continent", "en", default=None)
            death_date_str = None
        else:
            given = _safe_get(lau, "givenName", "en", default="")
            family = _safe_get(lau, "familyName", "en", default="")
            name = f"{given} {family}".strip()
            gender = lau.get("gender", "unknown")
            birth_date_str = _safe_get(lau, "birth", "date", default=None)
            birth_city = _safe_get(lau, "birth", "place", "city", "en", default=None)
            birth_country = _safe_get(lau, "birth", "place", "country", "en", default=None)
            birth_continent = _safe_get(lau, "birth", "place", "continent", "en", default=None)
            death_date_str = _safe_get(lau, "death", "date", default=None)

        birth_date = _parse_date(birth_date_str)
        death_date = _parse_date(death_date_str) if death_date_str else None

        # Parse each prize
        for prize in lau.get("nobelPrizes", []):
            category = _safe_get(prize, "category", "en", default="Unknown")
            award_year = prize.get("awardYear")
            motivation = _safe_get(prize, "motivation", "en", default="")
            prize_amount = prize.get("prizeAmount")
            prize_amount_adj = prize.get("prizeAmountAdjusted")
            portion = prize.get("portion", "1")
            date_awarded = prize.get("dateAwarded")

            # Affiliations
            affiliations = prize.get("affiliations", [])
            affiliation_name = None
            affiliation_country = None
            if affiliations and isinstance(affiliations, list) and len(affiliations) > 0:
                aff = affiliations[0]
                if isinstance(aff, dict):
                    affiliation_name = _safe_get(aff, "name", "en", default=None)
                    affiliation_country = _safe_get(aff, "country", "en", default=None)

            age_at_award = _compute_age(birth_date, award_year)
            lifespan = _compute_lifespan(birth_date, death_date)

            rows.append({
                "id": lau.get("id"),
                "name": name,
                "gender": gender,
                "is_org": is_org,
                "birth_date": birth_date,
                "death_date": death_date,
                "birth_city": birth_city,
                "birth_country": birth_country,
                "birth_continent": birth_continent,
                "category": category,
                "award_year": int(award_year) if award_year else None,
                "decade": (int(award_year) // 10) * 10 if award_year else None,
                "motivation": motivation if motivation else "",
                "prize_amount": prize_amount,
                "prize_amount_adjusted": prize_amount_adj,
                "portion": portion,
                "date_awarded": date_awarded,
                "affiliation": affiliation_name,
                "affiliation_country": affiliation_country,
                "age_at_award": age_at_award,
                "lifespan": lifespan,
                "is_alive": death_date is None and not is_org,
                "wiki_slug": lau.get("wikipedia", {}).get("slug"),
                "wikidata_id": lau.get("wikidata", {}).get("id"),
            })

    df = pd.DataFrame(rows)

    # Clean up
    df["award_year"] = pd.to_numeric(df["award_year"], errors="coerce").astype("Int64")
    df["decade"] = pd.to_numeric(df["decade"], errors="coerce").astype("Int64")
    df["age_at_award"] = pd.to_numeric(df["age_at_award"], errors="coerce").astype("Int64")
    df["prize_amount"] = pd.to_numeric(df["prize_amount"], errors="coerce")
    df["prize_amount_adjusted"] = pd.to_numeric(df["prize_amount_adjusted"], errors="coerce")

    # Sort by year
    df = df.sort_values("award_year").reset_index(drop=True)

    df = prefetch_wiki_images(df)
    df = prefetch_wikidata_nationalities(df)
    return df

def prefetch_wikidata_nationalities(df: pd.DataFrame) -> pd.DataFrame:
    """Batch fetch true Nationality (Country of Citizenship - P27) from Wikidata."""
    print("🌍 Prefetching real Nationalities from Wikidata...")
    import requests
    from concurrent.futures import ThreadPoolExecutor
    
    wikidata_ids = [s for s in df["wikidata_id"].dropna().unique()]
    id_to_country_qids = {}
    country_qids_to_fetch = set()
    
    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": "NobelPrize-Analytics-Dashboard/2.0 (research@example.com)"}
    
    # Step 1: Fetch P27 claims
    def fetch_claims_batch(batch):
        try:
            params = {"action": "wbgetentities", "ids": "|".join(batch), "props": "claims", "format": "json"}
            res = requests.get(url, params=params, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json().get("entities", {})
                for qid, entity_data in data.items():
                    claims = entity_data.get("claims", {}).get("P27", [])
                    qids = []
                    for c in claims:
                        try:
                            val_id = c["mainsnak"]["datavalue"]["value"]["id"]
                            qids.append(val_id)
                            country_qids_to_fetch.add(val_id)
                        except (KeyError, TypeError):
                            pass
                    id_to_country_qids[qid] = qids
        except Exception:
            pass
            
    batches = [wikidata_ids[i:i+50] for i in range(0, len(wikidata_ids), 50)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_claims_batch, batches)
        
    # Step 2: Fetch English labels for Country QIDs
    country_qid_list = list(country_qids_to_fetch)
    qid_to_label = {}
    
    def fetch_labels_batch(batch):
        try:
            params = {"action": "wbgetentities", "ids": "|".join(batch), "props": "labels", "languages": "en", "format": "json"}
            res = requests.get(url, params=params, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json().get("entities", {})
                for qid, entity_data in data.items():
                    try:
                        qid_to_label[qid] = entity_data["labels"]["en"]["value"]
                    except KeyError:
                        qid_to_label[qid] = qid
        except Exception:
            pass
            
    label_batches = [country_qid_list[i:i+50] for i in range(0, len(country_qid_list), 50)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_labels_batch, label_batches)
        
    # Step 3: Map nationalities back
    def map_nationalities(qid):
        if pd.isna(qid) or qid not in id_to_country_qids: return None
        country_qids = id_to_country_qids[qid]
        if not country_qids: return None
        labels = [qid_to_label.get(c, c) for c in country_qids]
        return ", ".join(labels)
        
    df["nationality"] = df["wikidata_id"].apply(map_nationalities)
    return df

def prefetch_wiki_images(df: pd.DataFrame) -> pd.DataFrame:
    """Batch fetch Wikipedia image URLs for all laureates."""
    print("📸 Prefetching Wikipedia portraits (fast batch mode)...")
    from concurrent.futures import ThreadPoolExecutor
    
    slugs = [s for s in df["wiki_slug"].unique() if pd.notna(s)]
    slug_to_url = {}
    
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "NobelPrize-Analytics-Dashboard/2.0 (student@example.com)"}
    
    def fetch_batch(batch):
        try:
            params = {
                "action": "query",
                "prop": "pageimages",
                "format": "json",
                "piprop": "thumbnail",
                "pithumbsize": 400,
                "titles": "|".join(batch),
                "redirects": 1
            }
            res = requests.get(url, params=params, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                pages = data.get("query", {}).get("pages", {})
                
                title_to_thumb = {}
                for page_data in pages.values():
                    if "thumbnail" in page_data:
                        title_to_thumb[page_data["title"]] = page_data["thumbnail"]["source"]
                
                title_map = {}
                for n in data.get("query", {}).get("normalized", []):
                    title_map[n["from"]] = n["to"]
                for r in data.get("query", {}).get("redirects", []):
                    title_map[r["from"]] = r["to"]
                    
                for slug in batch:
                    final_title = slug.replace("_", " ")
                    while final_title in title_map:
                        final_title = title_map[final_title]
                    if final_title in title_to_thumb:
                        slug_to_url[slug] = title_to_thumb[final_title]
        except Exception:
            pass
            
    batches = [slugs[i:i+50] for i in range(0, len(slugs), 50)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_batch, batches)
        
    df["img_url"] = df["wiki_slug"].apply(lambda s: slug_to_url.get(s, "") if pd.notna(s) else "")
    return df


def get_persons(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only person laureates (not organizations)."""
    return df[~df["is_org"]].copy()


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return high-level summary statistics."""
    persons = get_persons(df)
    return {
        "total_prizes": len(df),
        "unique_laureates": df["id"].nunique(),
        "persons": len(persons["id"].unique()),
        "organizations": df[df["is_org"]]["id"].nunique(),
        "categories": sorted(df["category"].unique().tolist()),
        "year_range": (int(df["award_year"].min()), int(df["award_year"].max())),
        "countries": df["birth_country"].nunique(),
        "avg_age": round(persons["age_at_award"].mean(), 1),
        "female_pct": round(
            len(persons[persons["gender"] == "female"]) / len(persons) * 100, 1
        ),
    }


if __name__ == "__main__":
    df = load_data()
    stats = get_summary_stats(df)
    print("\n📊 Nobel Prize Dataset Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")



