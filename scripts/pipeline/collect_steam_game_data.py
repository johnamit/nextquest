import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.client.steam_client import SteamClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-games-path", default="data/seed_games.csv")
    parser.add_argument("--raw-data-directory", default="data/raw/steam")
    parser.add_argument("--processed-data-directory", default="data/processed")
    parser.add_argument("--reviews-per-game", type=int, default=100)
    return parser.parse_args()


def load_seed_games(seed_games_path):
    seed_games_dataframe = pd.read_csv(seed_games_path)
    required_columns = {"app_id", "game_name"}
    missing_columns = required_columns.difference(seed_games_dataframe.columns)

    if missing_columns:
        missing_list = sorted(list(missing_columns))
        raise ValueError(f"Missing required columns in seed file: {missing_list}")

    return seed_games_dataframe.drop_duplicates(subset=["app_id"]).copy()


def get_price_gbp(game_payload):
    price_overview = game_payload.get("price_overview")
    if not price_overview:
        return None

    final_price_minor_units = price_overview.get("final")
    if final_price_minor_units is None:
        return None

    return final_price_minor_units / 100.0


def get_descriptions(items):
    descriptions = []
    for item in items:
        description_value = item.get("description")
        if description_value is not None:
            descriptions.append(description_value)

    return descriptions


def get_screenshot_urls(game_payload):
    screenshot_rows = game_payload.get("screenshots", [])
    screenshot_urls = []

    for screenshot_row in screenshot_rows:
        full_url = screenshot_row.get("path_full")
        if isinstance(full_url, str) and full_url.strip() != "":
            screenshot_urls.append(full_url)

    return screenshot_urls


def parse_game_row(app_id, app_details_payload, reviews_payload, current_players):
    game_payload = app_details_payload.get("data", {})
    review_summary_payload = reviews_payload.get("query_summary", {})

    total_reviews = review_summary_payload.get("total_reviews", 0)
    positive_reviews = review_summary_payload.get("total_positive", 0)

    if total_reviews > 0:
        positive_review_ratio = positive_reviews / total_reviews
    else:
        positive_review_ratio = None

    return {
        "app_id": app_id,
        "game_name": game_payload.get("name"),
        "release_date": game_payload.get("release_date", {}).get("date"),
        "is_free": game_payload.get("is_free"),
        "price_gbp": get_price_gbp(game_payload),
        "discount_percent": game_payload.get("price_overview", {}).get("discount_percent", 0),
        "genre_names": get_descriptions(game_payload.get("genres", [])),
        "category_names": get_descriptions(game_payload.get("categories", [])),
        "developer_names": game_payload.get("developers", []),
        "publisher_names": game_payload.get("publishers", []),
        "short_description": game_payload.get("short_description"),
        "header_image_url": game_payload.get("header_image"),
        "screenshot_urls": get_screenshot_urls(game_payload),
        "metacritic_score": game_payload.get("metacritic", {}).get("score"),
        "recommendations_total": game_payload.get("recommendations", {}).get("total"),
        "current_players": current_players,
        "total_reviews": total_reviews,
        "positive_reviews": positive_reviews,
        "negative_reviews": review_summary_payload.get("total_negative", 0),
        "positive_review_ratio": positive_review_ratio,
        "review_score_description": review_summary_payload.get("review_score_desc"),
    }


def parse_review_rows(app_id, reviews_payload):
    review_rows = []

    for review_payload in reviews_payload.get("reviews", []):
        author_payload = review_payload.get("author", {})
        review_rows.append(
            {
                "app_id": app_id,
                "recommendation_id": review_payload.get("recommendationid"),
                "review_text": review_payload.get("review"),
                "is_positive": review_payload.get("voted_up"),
                "votes_helpful": review_payload.get("votes_up"),
                "author_playtime_forever_minutes": author_payload.get("playtime_forever"),
            }
        )

    return review_rows


def normalize_reviews_dataframe(reviews_dataframe):
    working_dataframe = reviews_dataframe.copy()

    numeric_columns = [
        "app_id",
        "recommendation_id",
        "votes_helpful",
        "author_playtime_forever_minutes",
    ]

    for column_name in numeric_columns:
        if column_name in working_dataframe.columns:
            working_dataframe[column_name] = pd.to_numeric(working_dataframe[column_name], errors="coerce")

    if "is_positive" in working_dataframe.columns:
        working_dataframe["is_positive"] = working_dataframe["is_positive"].astype("boolean")

    return working_dataframe


def save_json_payload(file_path, payload):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)


def save_processed_outputs(games_dataframe, reviews_dataframe, processed_data_directory):
    processed_directory_path = Path(processed_data_directory)
    processed_directory_path.mkdir(parents=True, exist_ok=True)

    games_path = processed_directory_path / "games_catalog.parquet"
    reviews_path = processed_directory_path / "reviews_catalog.parquet"

    games_dataframe.to_parquet(games_path, index=False)
    normalize_reviews_dataframe(reviews_dataframe).to_parquet(reviews_path, index=False)


def main():
    load_dotenv()
    args = parse_args()

    steam_client = SteamClient()
    raw_data_directory = Path(args.raw_data_directory)

    seed_games_dataframe = load_seed_games(args.seed_games_path)

    game_rows = []
    review_rows = []

    for _, seed_row in tqdm(seed_games_dataframe.iterrows(), total=len(seed_games_dataframe), desc="Collecting Steam data"):
        app_id = int(seed_row["app_id"])

        app_details_payload = steam_client.get_app_details(app_id)
        if not app_details_payload.get("success"):
            continue

        reviews_payload = steam_client.get_reviews(app_id, number_of_reviews=args.reviews_per_game)
        current_players = steam_client.get_current_players(app_id)

        save_json_payload(raw_data_directory / "app_details" / f"{app_id}.json", app_details_payload)
        save_json_payload(raw_data_directory / "reviews" / f"{app_id}.json", reviews_payload)

        game_rows.append(parse_game_row(app_id, app_details_payload, reviews_payload, current_players))
        review_rows.extend(parse_review_rows(app_id, reviews_payload))

    games_dataframe = pd.DataFrame(game_rows)
    reviews_dataframe = pd.DataFrame(review_rows)
    save_processed_outputs(games_dataframe, reviews_dataframe, args.processed_data_directory)


if __name__ == "__main__":
    main()
