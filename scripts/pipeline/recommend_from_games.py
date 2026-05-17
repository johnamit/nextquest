import argparse
from pathlib import Path
import re

import faiss
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-directory", default="data/artifacts")
    parser.add_argument("--features-path", default="data/features/game_features.parquet")
    parser.add_argument("--liked-app-ids", default="")
    parser.add_argument("--liked-game-names", default="")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def parse_liked_app_ids(liked_app_ids_text):
    liked_app_ids = []

    if liked_app_ids_text.strip() == "":
        return liked_app_ids

    for app_id_text in liked_app_ids_text.split(","):
        cleaned_value = app_id_text.strip()
        if cleaned_value == "":
            continue
        liked_app_ids.append(int(cleaned_value))

    return liked_app_ids


def parse_liked_game_names(liked_game_names_text):
    liked_game_names = []

    if liked_game_names_text.strip() == "":
        return liked_game_names

    for game_name_text in liked_game_names_text.split(","):
        cleaned_value = game_name_text.strip()
        if cleaned_value == "":
            continue
        liked_game_names.append(cleaned_value.lower())

    return liked_game_names


def load_artifacts(artifacts_directory, features_path):
    artifacts_path = Path(artifacts_directory)

    metadata_path = artifacts_path / "game_genome_metadata.parquet"
    embeddings_path = artifacts_path / "game_genome_embeddings.npy"
    text_embeddings_path = artifacts_path / "game_text_embeddings.npy"
    image_embeddings_path = artifacts_path / "game_image_embeddings.npy"
    numeric_features_path = artifacts_path / "game_numeric_features.npy"
    faiss_index_path = artifacts_path / "game_genome_faiss.index"

    metadata_dataframe = pd.read_parquet(metadata_path)
    embeddings_matrix = np.load(embeddings_path)
    text_embeddings_matrix = np.load(text_embeddings_path)
    image_embeddings_matrix = np.load(image_embeddings_path)
    numeric_features_matrix = np.load(numeric_features_path)
    faiss_index = faiss.read_index(str(faiss_index_path))
    features_dataframe = pd.read_parquet(features_path)

    return (
        metadata_dataframe,
        embeddings_matrix,
        text_embeddings_matrix,
        image_embeddings_matrix,
        numeric_features_matrix,
        faiss_index,
        features_dataframe,
    )


def resolve_liked_indices(metadata_dataframe, liked_app_ids, liked_game_names):
    liked_index_set = set()

    if len(liked_app_ids) > 0:
        app_id_to_index = {}
        for row_index, app_id_value in enumerate(metadata_dataframe["app_id"].tolist()):
            app_id_to_index[int(app_id_value)] = row_index

        for app_id in liked_app_ids:
            if app_id in app_id_to_index:
                liked_index_set.add(app_id_to_index[app_id])

    if len(liked_game_names) > 0:
        game_name_to_index = {}
        for row_index, game_name_value in enumerate(metadata_dataframe["game_name"].fillna("").tolist()):
            game_name_to_index[str(game_name_value).lower()] = row_index

        for game_name in liked_game_names:
            if game_name in game_name_to_index:
                liked_index_set.add(game_name_to_index[game_name])

    return sorted(list(liked_index_set))


def build_query_vector(embeddings_matrix, liked_indices):
    liked_embeddings = embeddings_matrix[liked_indices]
    query_vector = np.mean(liked_embeddings, axis=0)

    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return query_vector.astype(np.float32)

    return (query_vector / query_norm).astype(np.float32)


def _normalize_vector(vector_value):
    vector_norm = np.linalg.norm(vector_value)
    if vector_norm == 0:
        return vector_value
    return vector_value / vector_norm


def _cosine_similarity(vector_a, vector_b):
    normalized_a = _normalize_vector(vector_a)
    normalized_b = _normalize_vector(vector_b)
    return float(np.dot(normalized_a, normalized_b))


def _to_text_list(value):
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    value_text = str(value).strip()
    if value_text.startswith("[") and value_text.endswith("]"):
        quoted_matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", value_text)
        extracted_values = []

        for left_match, right_match in quoted_matches:
            extracted_value = left_match if left_match != "" else right_match
            if extracted_value != "":
                extracted_values.append(extracted_value)

        if len(extracted_values) > 0:
            return extracted_values

        value_text = value_text[1:-1]

    if value_text == "":
        return []

    split_values = [part.strip().strip("'").strip('"') for part in value_text.split(",")]
    return [part for part in split_values if part != ""]


def _build_overlap_text(candidate_values, liked_values):
    candidate_set = set([value.lower() for value in candidate_values])
    liked_set = set([value.lower() for value in liked_values])
    overlap_values = sorted(list(candidate_set.intersection(liked_set)))

    if len(overlap_values) == 0:
        return ""

    readable_values = [value.title() for value in overlap_values[:3]]
    return ", ".join(readable_values)


def build_explanation(
    candidate_metadata_row,
    candidate_feature_row,
    liked_metadata_rows,
    liked_feature_rows,
    candidate_text_embedding,
    candidate_image_embedding,
    candidate_numeric_embedding,
    query_text_embedding,
    query_image_embedding,
    query_numeric_embedding,
):
    explanation_parts = []

    text_similarity = _cosine_similarity(candidate_text_embedding, query_text_embedding)
    image_similarity = _cosine_similarity(candidate_image_embedding, query_image_embedding)
    numeric_similarity = _cosine_similarity(candidate_numeric_embedding, query_numeric_embedding)

    if text_similarity >= image_similarity and text_similarity >= numeric_similarity:
        explanation_parts.append("it matches the same core gameplay/theme language as your liked games")
    elif image_similarity >= text_similarity and image_similarity >= numeric_similarity:
        explanation_parts.append("its visual style and atmosphere are close to what you already enjoy")
    else:
        explanation_parts.append("its gameplay and market profile align closely with your current taste")

    candidate_genres = _to_text_list(candidate_metadata_row.get("genre_names"))
    liked_genres = []
    for genre_values in liked_metadata_rows["genre_names"].tolist():
        liked_genres.extend(_to_text_list(genre_values))
    overlap_genres = _build_overlap_text(candidate_genres, liked_genres)
    if overlap_genres != "":
        explanation_parts.append(f"shared genre DNA: {overlap_genres}")

    candidate_categories = _to_text_list(candidate_metadata_row.get("category_names"))
    liked_categories = []
    for category_values in liked_metadata_rows["category_names"].tolist():
        liked_categories.extend(_to_text_list(category_values))
    overlap_categories = _build_overlap_text(candidate_categories, liked_categories)
    if overlap_categories != "":
        explanation_parts.append(f"similar play style tags: {overlap_categories}")

    candidate_developers = _to_text_list(candidate_metadata_row.get("developer_names"))
    liked_developers = []
    for developer_values in liked_metadata_rows["developer_names"].tolist():
        liked_developers.extend(_to_text_list(developer_values))
    overlap_developers = _build_overlap_text(candidate_developers, liked_developers)
    if overlap_developers != "":
        explanation_parts.append(f"made by a studio you already seem to trust: {overlap_developers}")

    candidate_audio_score = float(candidate_feature_row.get("strength_audio", 0) or 0)
    liked_audio_score = float(liked_feature_rows["strength_audio"].fillna(0).mean())
    if candidate_audio_score >= liked_audio_score and candidate_audio_score > 0:
        explanation_parts.append("reviews also highlight strong audio and soundtrack quality")

    candidate_sentiment = float(candidate_feature_row.get("positive_review_ratio", 0) or 0)
    liked_sentiment = float(liked_feature_rows["positive_review_ratio"].fillna(0).mean())
    if candidate_sentiment >= liked_sentiment:
        explanation_parts.append("player sentiment is as strong as your current favorites")

    preferred_theme_columns = [
        "theme_soulslike",
        "theme_dark_fantasy",
        "theme_fast_action",
        "theme_exploration",
        "theme_tactical",
    ]
    theme_name_map = {
        "theme_soulslike": "challenging soulslike combat",
        "theme_dark_fantasy": "dark fantasy tone",
        "theme_fast_action": "fast and responsive action",
        "theme_exploration": "strong exploration focus",
        "theme_tactical": "build and strategy depth",
    }

    liked_theme_profile = liked_feature_rows[preferred_theme_columns].fillna(0).mean()
    best_theme = None
    best_theme_alignment = -1

    for theme_column in preferred_theme_columns:
        candidate_value = float(candidate_feature_row.get(theme_column, 0) or 0)
        liked_value = float(liked_theme_profile.get(theme_column, 0) or 0)
        alignment_value = min(candidate_value, liked_value)

        if alignment_value > best_theme_alignment:
            best_theme_alignment = alignment_value
            best_theme = theme_column

    if best_theme is not None and best_theme_alignment > 0:
        explanation_parts.append(f"it fits your preference for {theme_name_map[best_theme]}")

    if len(explanation_parts) > 2:
        explanation_parts = explanation_parts[:2]

    return "; ".join(explanation_parts)


def _format_genre_text(primary_genre_value):
    genre_values = _to_text_list(primary_genre_value)
    if len(genre_values) == 0:
        return "Unknown"

    return ", ".join([genre_value.title() for genre_value in genre_values[:3]])


def recommend_games(
    metadata_dataframe,
    embeddings_matrix,
    text_embeddings_matrix,
    image_embeddings_matrix,
    numeric_features_matrix,
    faiss_index,
    features_dataframe,
    liked_indices,
    top_k,
):
    query_vector = build_query_vector(embeddings_matrix, liked_indices)
    query_vector = np.expand_dims(query_vector, axis=0)

    search_count = min(len(metadata_dataframe), max(top_k + len(liked_indices) + 10, top_k))
    similarity_scores, neighbor_indices = faiss_index.search(query_vector, search_count)

    liked_app_id_set = set(metadata_dataframe.iloc[liked_indices]["app_id"].tolist())
    liked_rows = features_dataframe[features_dataframe["app_id"].isin(liked_app_id_set)].copy()
    liked_metadata_rows = metadata_dataframe[metadata_dataframe["app_id"].isin(liked_app_id_set)].copy()

    query_text_embedding = np.mean(text_embeddings_matrix[liked_indices], axis=0)
    query_image_embedding = np.mean(image_embeddings_matrix[liked_indices], axis=0)
    query_numeric_embedding = np.mean(numeric_features_matrix[liked_indices], axis=0)

    recommendation_rows = []

    for position in range(len(neighbor_indices[0])):
        neighbor_index = int(neighbor_indices[0][position])
        similarity_score = float(similarity_scores[0][position])

        candidate_metadata_row = metadata_dataframe.iloc[neighbor_index]
        candidate_app_id = int(candidate_metadata_row["app_id"])

        if candidate_app_id in liked_app_id_set:
            continue

        candidate_feature_rows = features_dataframe[features_dataframe["app_id"] == candidate_app_id]
        if candidate_feature_rows.empty:
            continue

        candidate_feature_row = candidate_feature_rows.iloc[0]
        explanation_text = build_explanation(
            candidate_metadata_row=candidate_metadata_row,
            candidate_feature_row=candidate_feature_row,
            liked_metadata_rows=liked_metadata_rows,
            liked_feature_rows=liked_rows,
            candidate_text_embedding=text_embeddings_matrix[neighbor_index],
            candidate_image_embedding=image_embeddings_matrix[neighbor_index],
            candidate_numeric_embedding=numeric_features_matrix[neighbor_index],
            query_text_embedding=query_text_embedding,
            query_image_embedding=query_image_embedding,
            query_numeric_embedding=query_numeric_embedding,
        )

        recommendation_rows.append(
            {
                "app_id": candidate_app_id,
                "game_name": candidate_metadata_row["game_name"],
                "primary_genre": candidate_feature_row.get("primary_genre"),
                "header_image_url": candidate_metadata_row.get("header_image_url"),
                "price_gbp": candidate_feature_row.get("price_gbp"),
                "positive_review_ratio": candidate_feature_row.get("positive_review_ratio"),
                "current_players": candidate_feature_row.get("current_players"),
                "metacritic_score": candidate_feature_row.get("metacritic_score"),
                "cosine_similarity": similarity_score,
                "why_recommended": explanation_text,
            }
        )

        if len(recommendation_rows) >= top_k:
            break

    return pd.DataFrame(recommendation_rows)


def generate_recommendations(
    artifacts_directory,
    features_path,
    liked_app_ids_text,
    liked_game_names_text,
    top_k,
):
    liked_app_ids = parse_liked_app_ids(liked_app_ids_text)
    liked_game_names = parse_liked_game_names(liked_game_names_text)

    (
        metadata_dataframe,
        embeddings_matrix,
        text_embeddings_matrix,
        image_embeddings_matrix,
        numeric_features_matrix,
        faiss_index,
        features_dataframe,
    ) = load_artifacts(
        artifacts_directory,
        features_path,
    )

    liked_indices = resolve_liked_indices(
        metadata_dataframe=metadata_dataframe,
        liked_app_ids=liked_app_ids,
        liked_game_names=liked_game_names,
    )

    if len(liked_indices) == 0:
        raise ValueError("No liked games matched by app ID or game name. Provide at least one valid liked game.")

    recommendations_dataframe = recommend_games(
        metadata_dataframe=metadata_dataframe,
        embeddings_matrix=embeddings_matrix,
        text_embeddings_matrix=text_embeddings_matrix,
        image_embeddings_matrix=image_embeddings_matrix,
        numeric_features_matrix=numeric_features_matrix,
        faiss_index=faiss_index,
        features_dataframe=features_dataframe,
        liked_indices=liked_indices,
        top_k=top_k,
    )

    return recommendations_dataframe


def print_recommendations(recommendations_dataframe):
    if recommendations_dataframe.empty:
        print("No recommendations could be generated with the provided liked games.")
        return

    for rank_index, (_, recommendation_row) in enumerate(recommendations_dataframe.iterrows(), start=1):
        print("-" * 100)
        print(f"{rank_index}. {recommendation_row['game_name']} (AppID: {int(recommendation_row['app_id'])})")
        print(f"   Similarity: {float(recommendation_row['cosine_similarity']):.4f}")
        print(f"   Genre: {_format_genre_text(recommendation_row.get('primary_genre', 'Unknown'))}")

        price_value = recommendation_row.get("price_gbp")
        if price_value is not None and not pd.isna(price_value):
            print(f"   Price: GBP {float(price_value):.2f}")
        else:
            print("   Price: N/A")

        sentiment_value = recommendation_row.get("positive_review_ratio")
        if sentiment_value is not None and not pd.isna(sentiment_value):
            print(f"   Positive Review Ratio: {float(sentiment_value):.1%}")
        else:
            print("   Positive Review Ratio: N/A")

        current_players_value = recommendation_row.get("current_players")
        if current_players_value is not None and not pd.isna(current_players_value):
            print(f"   Current Players: {int(float(current_players_value)):,}")
        else:
            print("   Current Players: N/A")

        print(f"   Why: {recommendation_row['why_recommended']}")


def main():
    args = parse_args()
    recommendations_dataframe = generate_recommendations(
        args.artifacts_directory,
        args.features_path,
        args.liked_app_ids,
        args.liked_game_names,
        args.top_k,
    )

    print_recommendations(recommendations_dataframe)


if __name__ == "__main__":
    main()
