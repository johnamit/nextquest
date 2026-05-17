import argparse
import ast
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-path", default="data/processed/games_catalog.parquet")
    parser.add_argument("--reviews-path", default="data/processed/reviews_catalog.parquet")
    parser.add_argument("--features-path", default="data/features/game_features.parquet")
    return parser.parse_args()


def _to_text_list(value):
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value == "":
            return []

        if stripped_value.startswith("[") and stripped_value.endswith("]"):
            try:
                parsed_value = ast.literal_eval(stripped_value)
                if isinstance(parsed_value, list):
                    return [str(item) for item in parsed_value if item is not None]
            except (ValueError, SyntaxError):
                pass

        return [stripped_value]

    return [str(value)]


def _extract_primary_genre(genre_names_value):
    genre_names = _to_text_list(genre_names_value)
    if len(genre_names) == 0:
        return "Unknown"

    return genre_names[0]


def _safe_series_divide(numerator_series, denominator_series):
    return numerator_series.divide(denominator_series).replace([float("inf"), -float("inf")], pd.NA)


def build_review_text_signals(reviews_dataframe):
    issue_keywords = {
        "weakness_content": ["content", "shallow", "repetitive", "endgame", "grind"],
        "weakness_performance": ["performance", "fps", "stutter", "lag", "optimization", "crash"],
        "weakness_controls": ["controls", "ui", "clunky", "camera", "menu"],
    }

    strength_keywords = {
        "strength_gameplay": ["gameplay", "fun", "addictive", "mechanics", "combat"],
        "strength_story": ["story", "characters", "writing", "world", "lore"],
        "strength_visuals": ["graphics", "art", "visual", "music", "atmosphere"],
        "strength_audio": ["soundtrack", "music", "audio", "sound design", "voice acting"],
    }

    theme_keywords = {
        "theme_soulslike": ["soulslike", "boss", "parry", "dodge", "stamina", "punishing"],
        "theme_exploration": ["explore", "open world", "discovery", "map", "hidden"],
        "theme_fast_action": ["fast-paced", "action", "reflex", "combo", "responsive"],
        "theme_dark_fantasy": ["dark", "grim", "gothic", "fantasy", "bleak"],
        "theme_coop_social": ["co-op", "coop", "friends", "multiplayer", "team"],
        "theme_tactical": ["tactical", "strategy", "planning", "build", "loadout"],
        "theme_horror_tension": ["horror", "scary", "tense", "fear", "atmosphere"],
    }

    working_reviews_dataframe = reviews_dataframe.copy()
    working_reviews_dataframe["review_text"] = working_reviews_dataframe["review_text"].fillna("").str.lower()

    review_counts = working_reviews_dataframe.groupby("app_id")["recommendation_id"].count().rename("review_sample_size")
    feature_dataframe = pd.DataFrame(index=review_counts.index)
    feature_dataframe["review_sample_size"] = review_counts

    for feature_name, keyword_list in issue_keywords.items():
        keyword_pattern = "|".join(keyword_list)
        matching_rows = working_reviews_dataframe["review_text"].str.contains(keyword_pattern, regex=True)
        matching_counts = working_reviews_dataframe[matching_rows].groupby("app_id")["recommendation_id"].count()
        feature_dataframe[feature_name] = _safe_series_divide(
            matching_counts.reindex(feature_dataframe.index).fillna(0),
            review_counts,
        )

    for feature_name, keyword_list in strength_keywords.items():
        keyword_pattern = "|".join(keyword_list)
        matching_rows = working_reviews_dataframe["review_text"].str.contains(keyword_pattern, regex=True)
        matching_counts = working_reviews_dataframe[matching_rows].groupby("app_id")["recommendation_id"].count()
        feature_dataframe[feature_name] = _safe_series_divide(
            matching_counts.reindex(feature_dataframe.index).fillna(0),
            review_counts,
        )

    for feature_name, keyword_list in theme_keywords.items():
        keyword_pattern = "|".join(keyword_list)
        matching_rows = working_reviews_dataframe["review_text"].str.contains(keyword_pattern, regex=True)
        matching_counts = working_reviews_dataframe[matching_rows].groupby("app_id")["recommendation_id"].count()
        feature_dataframe[feature_name] = _safe_series_divide(
            matching_counts.reindex(feature_dataframe.index).fillna(0),
            review_counts,
        )

    high_playtime_reviews = working_reviews_dataframe[
        working_reviews_dataframe["author_playtime_forever_minutes"].fillna(0) >= 600
    ].copy()
    high_playtime_sentiment = high_playtime_reviews.groupby("app_id")["is_positive"].mean()
    feature_dataframe["high_playtime_sentiment"] = high_playtime_sentiment.reindex(feature_dataframe.index).fillna(0)

    return feature_dataframe.reset_index().fillna(0)


def build_recommender_features(games_dataframe, reviews_dataframe):
    features_dataframe = games_dataframe.copy()

    features_dataframe["primary_genre"] = features_dataframe["genre_names"].apply(_extract_primary_genre)
    features_dataframe["market_player_review_ratio"] = _safe_series_divide(
        features_dataframe["current_players"],
        features_dataframe["total_reviews"],
    )
    features_dataframe["price_gbp"] = pd.to_numeric(features_dataframe["price_gbp"], errors="coerce")
    features_dataframe["positive_review_ratio"] = pd.to_numeric(features_dataframe["positive_review_ratio"], errors="coerce")

    review_feature_dataframe = build_review_text_signals(reviews_dataframe)
    features_dataframe = features_dataframe.merge(review_feature_dataframe, on="app_id", how="left")

    fill_zero_columns = [
        "market_player_review_ratio",
        "review_sample_size",
        "weakness_content",
        "weakness_performance",
        "weakness_controls",
        "strength_gameplay",
        "strength_story",
        "strength_visuals",
        "strength_audio",
        "theme_soulslike",
        "theme_exploration",
        "theme_fast_action",
        "theme_dark_fantasy",
        "theme_coop_social",
        "theme_tactical",
        "theme_horror_tension",
        "high_playtime_sentiment",
    ]

    for column_name in fill_zero_columns:
        if column_name in features_dataframe.columns:
            features_dataframe[column_name] = features_dataframe[column_name].fillna(0)

    return features_dataframe


def main():
    args = parse_args()

    games_dataframe = pd.read_parquet(args.games_path)
    reviews_dataframe = pd.read_parquet(args.reviews_path)

    features_dataframe = build_recommender_features(games_dataframe, reviews_dataframe)

    output_path = Path(args.features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_dataframe.to_parquet(output_path, index=False)
    features_dataframe.to_csv(output_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
