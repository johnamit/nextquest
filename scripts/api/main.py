from ast import literal_eval
import math
from pathlib import Path
import re

import os
import shutil
from huggingface_hub import snapshot_download

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd

from scripts.client.steam_client import SteamClient
from scripts.pipeline.recommend_from_games import generate_recommendations

load_dotenv()

REQUIRED_DATA_FILES = [
    "data/artifacts/game_genome_metadata.parquet",
    "data/features/game_features.parquet",
    "data/processed/games_catalog.parquet",
    "data/processed/reviews_catalog.parquet",
]

app = FastAPI(title="NextQuest Recommendation System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_required_data_files() -> None:
    missing_paths = [file_path for file_path in REQUIRED_DATA_FILES if not Path(file_path).exists()]
    if missing_paths:
        raise RuntimeError(f"Missing required data files: {missing_paths}")


def sync_data_from_hf() -> None:
    hf_repo_id = os.getenv("HF_REPO_ID", "").strip()
    hf_repo_type = os.getenv("HF_REPO_TYPE", "dataset").strip()
    hf_revision = os.getenv("HF_REVISION", "").strip() or None

    if hf_repo_id == "":
        raise RuntimeError("HF_REPO_ID is required for startup data sync.")

    snapshot_path = snapshot_download(
        repo_id=hf_repo_id,
        repo_type=hf_repo_type,
        revision=hf_revision,
        local_dir=None,
    )

    source_data_dir = Path(snapshot_path) / "data"
    if not source_data_dir.exists():
        raise RuntimeError(f"HF snapshot missing top-level data/ folder: {source_data_dir}")

    target_data_dir = Path("data")
    if target_data_dir.exists():
        shutil.rmtree(target_data_dir)
    shutil.copytree(source_data_dir, target_data_dir)

    validate_required_data_files()


@app.on_event("startup")
def startup_data_sync():
    sync_data_from_hf()


# Cleaning for JSON helper functions
def clean_for_json(value):
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return None
        return numeric_value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if pd.isna(value):
        return None

    return value

def dataframe_to_json_records(dataframe: pd.DataFrame):
    cleaned_dataframe = dataframe.replace([np.inf, -np.inf], pd.NA)
    cleaned_dataframe = cleaned_dataframe.where(pd.notna(cleaned_dataframe), None)
    records = cleaned_dataframe.to_dict(orient="records")
    return clean_for_json(records)


# Data loading helper functions
def load_games_catalog(games_path: str):
    catalog_path = Path(games_path)
    if not catalog_path.exists():
        return pd.DataFrame()
    
    games_dataframe = pd.read_parquet(catalog_path)
    games_dataframe = games_dataframe.dropna(subset=["app_id", "game_name"]).copy()
    games_dataframe["app_id"] = pd.to_numeric(games_dataframe["app_id"], errors="coerce")
    games_dataframe = games_dataframe.dropna(subset=["app_id"]).copy()
    games_dataframe["app_id"] = games_dataframe["app_id"].astype("int64")
    games_dataframe = games_dataframe.drop_duplicates(subset=["app_id"]).copy()
    games_dataframe = games_dataframe.sort_values(by="game_name").reset_index(drop=True)
    return games_dataframe

def load_features_dataframe(features_path: str):
    path = Path(features_path)
    if not path.exists():
        return pd.DataFrame()
    
    features_df = pd.read_parquet(path)
    features_df = features_df.dropna(subset=["app_id"]).copy()
    features_df["app_id"] = pd.to_numeric(features_df["app_id"], errors="coerce")
    features_df = features_df.dropna(subset=["app_id"]).copy()
    features_df["app_id"] = features_df["app_id"].astype("int64")
    features_df = features_df.drop_duplicates(subset=["app_id"]).copy()
    return features_df

def load_reviews_dataframe(reviews_path: str):
    path = Path(reviews_path)
    if not path.exists():
        return pd.DataFrame()
    
    reviews_df = pd.read_parquet(path)
    if "app_id" not in reviews_df.columns:
        return pd.DataFrame()
    
    reviews_df["app_id"] = pd.to_numeric(reviews_df["app_id"], errors="coerce")
    reviews_df = reviews_df.dropna(subset=["app_id"]).copy()
    reviews_df["app_id"] = reviews_df["app_id"].astype("int64")
    
    if "recommendation_id" in reviews_df.columns:
        reviews_df["recommendation_id"] = pd.to_numeric(reviews_df["recommendation_id"], errors="coerce")
    
    return reviews_df

def load_known_app_ids(artifacts_directory: str):
    metadata_path = Path(artifacts_directory) / "game_genome_metadata.parquet"
    metadata_df = pd.read_parquet(metadata_path)
    metadata_df["app_id"] = pd.to_numeric(metadata_df["app_id"], errors="coerce")
    metadata_df = metadata_df.dropna(subset=["app_id"]).copy()
    metadata_df["app_id"] = metadata_df["app_id"].astype("int64")
    return set(metadata_df["app_id"].tolist())


# Steam client helper functions
def fetch_owned_games_dataframe(steam_client: SteamClient, steam_id: str):
    payload = steam_client.get_owned_games(steam_id)
    games = payload.get("games", [])
    if not games:
        return pd.DataFrame(columns=["app_id", "playtime_forever", "steam_game_name"])
    
    owned_df = pd.DataFrame(games)
    owned_df["app_id"] = pd.to_numeric(owned_df.get("appid"), errors="coerce")
    owned_df["playtime_forever"] = pd.to_numeric(owned_df.get("playtime_forever"), errors="coerce").fillna(0)
    owned_df = owned_df.rename(columns={"name": "steam_game_name"})
    owned_df = owned_df.dropna(subset=["app_id"]).copy()
    owned_df["app_id"] = owned_df["app_id"].astype("int64")
    owned_df = owned_df.drop_duplicates(subset=["app_id"]).copy()
    return owned_df

def select_liked_app_ids(owned_df: pd.DataFrame, known_app_ids: set[int], max_liked_games: int):
    matched_df = owned_df[owned_df["app_id"].isin(known_app_ids)].copy()
    matched_df = matched_df.sort_values(by="playtime_forever", ascending=False)
    liked_app_ids = matched_df["app_id"].head(max_liked_games).tolist()
    return liked_app_ids, matched_df


# Parsing and formatting helper functions
def parse_liked_app_ids_text(liked_app_ids_text: str) -> list[int]:
    if liked_app_ids_text.strip() == "":
        return []
    
    values = []
    for raw_value in liked_app_ids_text.split(","):
        clean_value = raw_value.strip()
        if clean_value == "":
            continue
        if clean_value.isdigit():
            values.append(int(clean_value))
    return values


def to_text_list(value):
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [str(item).strip() for item in value if item is not None and str(item).strip() != ""]
    
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    
    if not isinstance(value, str):
        value = str(value)
    
    stripped = value.strip()
    if stripped == "":
        return []
    
    # If bracketed list-like string, parse robustly
    if stripped.startswith("[") and stripped.endswith("]"):
        # literal_eval (handles valid Python list strings)
        try:
            parsed = literal_eval(stripped)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item is not None and str(item).strip() != ""]
        except Exception:
            pass

        # Extract quoted values: 'Action' or "Action"
        inner = stripped[1:-1].strip()
        quoted_parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", inner)
        flattened = []
        for left, right in quoted_parts:
            token = (left or right).strip()
            if token != "":
                flattened.append(token)
        if flattened:
            return flattened
        
        # Comma fallback
        if "," in inner:
            parts = [part.strip().strip("'").strip('"') for part in inner.split(",")]
            parts = [part for part in parts if part != ""]
            if parts:
                return parts
        
        # 4) Whitespace fallback
        parts = [part.strip().strip("'").strip('"') for part in inner.split()]
        parts = [part for part in parts if part != ""]
        if parts:
            return parts

    return [stripped] # fallback to single value list


def normalize_text_values(values):
    normalized = []

    for raw in values:
        text = str(raw).strip()
        if text == "":
            continue

        # unwrap list-like wrappers if present
        text = text.strip("[]").strip()
        if text == "":
            continue

        # split quoted chunks if they exist (e.g. 'Action' 'Adventure')
        quoted_parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        if quoted_parts:
            for left_value, right_value in quoted_parts:
                token = (left_value or right_value).strip()
                if token != "":
                    normalized.append(token)
            continue

        # comma separated fallback
        if "," in text:
            for part in text.split(","):
                token = part.strip().strip("'").strip('"').strip()
                if token != "":
                    normalized.append(token)
            continue

        # plain token fallback
        clean_text = text.strip("'").strip('"').strip()
        if clean_text != "":
            normalized.append(clean_text)

    # dedupe while keeping order
    seen = set()
    unique = []
    for item in normalized:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique

def extract_year(release_date_value):
    if release_date_value is None or pd.isna(release_date_value):
        return None
    text = str(release_date_value)
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return None
    return int(match.group(0))

def build_reviews_list(reviews_df: pd.DataFrame, app_id: int, max_reviews: int = 10) -> list[str]:
    if reviews_df.empty:
        return []
    game_reviews = reviews_df[reviews_df["app_id"] == app_id].copy()
    if game_reviews.empty:
        return []
    if "review_text" not in game_reviews.columns:
        return []
    game_reviews["review_text"] = game_reviews["review_text"].fillna("").astype(str).str.strip()
    game_reviews = game_reviews[game_reviews["review_text"] != ""].copy()
    if game_reviews.empty:
        return []
    if "recommendation_id" in game_reviews.columns:
        game_reviews = game_reviews.sort_values(by="recommendation_id", ascending=False, na_position="last")
    return game_reviews["review_text"].head(max_reviews).tolist()


# API endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/games")
def games(games_path: str = "data/processed/games_catalog.parquet"):
    games_dataframe = load_games_catalog(games_path)
    if games_dataframe.empty:
        return {"count": 0, "data": []}
    rows = games_dataframe[["app_id", "game_name", "header_image_url"]].to_dict(orient="records")
    return clean_for_json({"count": len(rows), "data": rows})

@app.get("/steam/library")
def steam_library(
    steam_id: str = "",
    games_path: str = "data/processed/games_catalog.parquet",
):
    if steam_id.strip() == "":
        return clean_for_json({
            "mode": "steam_library",
            "steam_id": steam_id,
            "message": "steam_id is required.",
            "count": 0,
            "data": [],
        })
    steam_client = SteamClient()
    owned_df = fetch_owned_games_dataframe(steam_client, steam_id)
    if owned_df.empty:
        return clean_for_json({
            "mode": "steam_library",
            "steam_id": steam_id,
            "message": "Could not access Steam profile. Check SteamID and privacy settings.",
            "count": 0,
            "data": [],
        })
    catalog_df = load_games_catalog(games_path)
    if catalog_df.empty:
        owned_rows = owned_df[["app_id", "steam_game_name", "playtime_forever"]].copy()
        owned_rows = owned_rows.rename(columns={"steam_game_name": "game_name"})
        owned_rows["header_image_url"] = None
        rows = owned_rows[["app_id", "game_name", "header_image_url", "playtime_forever"]].to_dict(orient="records")
        return clean_for_json({
            "mode": "steam_library",
            "steam_id": steam_id,
            "count": len(rows),
            "data": rows,
        })
    merged = owned_df.merge(
        catalog_df[["app_id", "game_name", "header_image_url"]],
        on="app_id",
        how="left",
    )
    merged["game_name"] = merged["game_name"].fillna(merged.get("steam_game_name", "Unknown"))
    merged = merged.sort_values(by="playtime_forever", ascending=False)
    rows = merged[["app_id", "game_name", "header_image_url", "playtime_forever"]].to_dict(orient="records")
    return clean_for_json({
        "mode": "steam_library",
        "steam_id": steam_id,
        "count": len(rows),
        "data": rows,
    })


@app.get("/recommendations")
def recommendations(
    liked_app_ids: str = "",
    liked_game_names: str = "",
    steam_id: str = "",
    top_k: int = 10,
    exclude_owned: bool = True,
    max_liked_games: int = 20,
    artifacts_directory: str = "data/artifacts",
    features_path: str = "data/features/game_features.parquet",
):
    owned_app_ids: set[int] = set()
    recommendation_top_k = top_k

    if exclude_owned and steam_id.strip() != "":
        try:
            steam_client = SteamClient()
            owned_df = fetch_owned_games_dataframe(steam_client, steam_id)
            if not owned_df.empty:
                owned_app_ids = set(owned_df["app_id"].astype("int64").tolist())
                recommendation_top_k = top_k + 50
        except Exception:
            owned_app_ids = set()

    def finalize_recommendations(recommendations_df: pd.DataFrame):
        if exclude_owned and len(owned_app_ids) > 0 and not recommendations_df.empty:
            recommendations_df = recommendations_df[~recommendations_df["app_id"].isin(owned_app_ids)].copy()
        return recommendations_df.head(top_k).reset_index(drop=True)

    # Mode 1: Manual by app ids (primary path for frontend)
    liked_app_id_values = parse_liked_app_ids_text(liked_app_ids)
    if len(liked_app_id_values) > 0:
        liked_app_ids_text = ",".join(str(app_id) for app_id in liked_app_id_values)
        recommendations_df = generate_recommendations(
            artifacts_directory=artifacts_directory,
            features_path=features_path,
            liked_app_ids_text=liked_app_ids_text,
            liked_game_names_text="",
            top_k=recommendation_top_k,
        )
        recommendations_df = finalize_recommendations(recommendations_df)
        return clean_for_json({
            "mode": "manual",
            "count": len(recommendations_df),
            "data": dataframe_to_json_records(recommendations_df),
        })
    
    # Mode 2: Manual by game names
    if liked_game_names.strip() != "":
        recommendations_df = generate_recommendations(
            artifacts_directory=artifacts_directory,
            features_path=features_path,
            liked_app_ids_text="",
            liked_game_names_text=liked_game_names,
            top_k=recommendation_top_k,
        )
        recommendations_df = finalize_recommendations(recommendations_df)
        return clean_for_json({
            "mode": "manual",
            "count": len(recommendations_df),
            "data": dataframe_to_json_records(recommendations_df),
        })
    
    # Mode 3: Steam library
    if steam_id.strip() == "":
        return clean_for_json({
            "mode": "steam",
            "message": "Provide one of liked_app_ids, liked_game_names, or steam_id.",
            "count": 0,
            "data": [],
        })
    
    steam_client = SteamClient()
    
    owned_df = fetch_owned_games_dataframe(steam_client, steam_id)
    if owned_df.empty:
        return clean_for_json({
            "mode": "steam",
            "message": "Could not access Steam profile. Check SteamID and privacy settings.",
            "count": 0,
            "data": [],
        })
    
    known_app_ids = load_known_app_ids(artifacts_directory)
    
    liked_app_ids_list, matched_df = select_liked_app_ids(owned_df, known_app_ids, max_liked_games)
    if len(liked_app_ids_list) == 0:
        return clean_for_json({
            "mode": "steam",
            "message": "No overlap between Steam library and local model catalog.",
            "owned_count": len(owned_df),
            "matched_count": 0,
            "seed_count": 0,
            "count": 0,
            "data": [],
        })
    liked_app_ids_text = ",".join(str(app_id) for app_id in liked_app_ids_list)
    
    recommendations_df = generate_recommendations(
        artifacts_directory=artifacts_directory,
        features_path=features_path,
        liked_app_ids_text=liked_app_ids_text,
        liked_game_names_text="",
        top_k=recommendation_top_k,
    )
    recommendations_df = finalize_recommendations(recommendations_df)
    
    return clean_for_json({
        "mode": "steam",
        "owned_count": len(owned_df),
        "matched_count": len(matched_df),
        "seed_count": len(liked_app_ids_list),
        "count": len(recommendations_df),
        "data": dataframe_to_json_records(recommendations_df),
    })


@app.get("/game-profile")
def game_profile(
    app_id: int,
    games_path: str = "data/processed/games_catalog.parquet",
    features_path: str = "data/features/game_features.parquet",
    reviews_path: str = "data/processed/reviews_catalog.parquet",
):
    games_df = load_games_catalog(games_path)
    features_df = load_features_dataframe(features_path)
    reviews_df = load_reviews_dataframe(reviews_path)
    
    game_row = pd.Series(dtype="object")
    feature_row = pd.Series(dtype="object")
    
    if not games_df.empty:
        matches = games_df[games_df["app_id"] == app_id]
        if not matches.empty:
            game_row = matches.iloc[0]
    
    if not features_df.empty:
        matches = features_df[features_df["app_id"] == app_id]
        if not matches.empty:
            feature_row = matches.iloc[0]
    
    if game_row.empty and feature_row.empty:
        return clean_for_json({
            "message": f"No game profile found for app_id={app_id}.",
            "app_id": app_id,
            "game_name": None,
            "year": None,
            "studio": None,
            "genres": [],
            "main_genres": [],
            "description": None,
            "header_image_url": None,
            "screenshot_urls": [],
            "price_gbp": None,
            "current_players": None,
            "metacritic_score": None,
            "positive_review_ratio": None,
            "positive_reviews": None,
            "negative_reviews": None,
            "reviews": [],
        })
    
    # Prefer feature row values, fallback to game row values
    game_name = feature_row.get("game_name") if not feature_row.empty else None
    if not game_name and not game_row.empty:
        game_name = game_row.get("game_name")
    
    header_image_url = feature_row.get("header_image_url") if not feature_row.empty else None
    if not header_image_url and not game_row.empty:
        header_image_url = game_row.get("header_image_url")
    
    release_date_value = game_row.get("release_date") if not game_row.empty else None
    if release_date_value is None and not feature_row.empty:
        release_date_value = feature_row.get("release_date")
    
    year = extract_year(release_date_value)
    genre_values = []
    
    if not feature_row.empty and "genre_names" in feature_row:
        genre_values = to_text_list(feature_row.get("genre_names"))
    elif not game_row.empty and "genre_names" in game_row:
        genre_values = to_text_list(game_row.get("genre_names"))
    
    genre_values = normalize_text_values(genre_values)
    main_genres = genre_values[:2]

    studio_values = []
    if not game_row.empty and "developer_names" in game_row:
        studio_values = to_text_list(game_row.get("developer_names"))
    if len(studio_values) == 0 and not feature_row.empty and "developer_names" in feature_row:
        studio_values = to_text_list(feature_row.get("developer_names"))
    studio_values = normalize_text_values(studio_values)
    studio = studio_values[0] if len(studio_values) > 0 else None

    description = None
    if not game_row.empty and "short_description" in game_row:
        description = game_row.get("short_description")
    if (description is None or str(description).strip() == "") and not feature_row.empty and "short_description" in feature_row:
        description = feature_row.get("short_description")
    if description is not None and str(description).strip() == "":
        description = None

    screenshot_urls = []
    if not game_row.empty and "screenshot_urls" in game_row:
        screenshot_urls = to_text_list(game_row.get("screenshot_urls"))
    if len(screenshot_urls) == 0 and not feature_row.empty and "screenshot_urls" in feature_row:
        screenshot_urls = to_text_list(feature_row.get("screenshot_urls"))
    screenshot_urls = [url for url in screenshot_urls if isinstance(url, str) and url.strip() != ""]
    screenshot_urls = list(dict.fromkeys(screenshot_urls))[:8]

    price_gbp = feature_row.get("price_gbp") if not feature_row.empty else None
    current_players = feature_row.get("current_players") if not feature_row.empty else None
    metacritic_score = feature_row.get("metacritic_score") if not feature_row.empty else None
    
    positive_review_ratio = feature_row.get("positive_review_ratio") if not feature_row.empty else None
    positive_reviews = feature_row.get("positive_reviews") if not feature_row.empty else None
    negative_reviews = feature_row.get("negative_reviews") if not feature_row.empty else None
    reviews = build_reviews_list(reviews_df, app_id=app_id, max_reviews=10)
    
    response_payload = {
        "app_id": app_id,
        "game_name": game_name,
        "year": year,
        "studio": studio,
        "genres": genre_values,
        "main_genres": main_genres,
        "description": description,
        "header_image_url": header_image_url,
        "screenshot_urls": screenshot_urls,
        "price_gbp": price_gbp,
        "current_players": current_players,
        "metacritic_score": metacritic_score,
        "positive_review_ratio": positive_review_ratio,
        "positive_reviews": positive_reviews,
        "negative_reviews": negative_reviews,
        "reviews": reviews,
    }
    return clean_for_json(response_payload)
