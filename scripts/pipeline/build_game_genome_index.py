import argparse
import ast
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default="data/features/game_features.parquet")
    parser.add_argument("--artifacts-directory", default="data/artifacts")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--image-embedding-model-name", default="clip-ViT-B-32")
    parser.add_argument("--images-per-game", type=int, default=3)
    parser.add_argument("--similar-games-count", type=int, default=10)
    return parser.parse_args()


def _to_text_list(value):
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]

    if value is None or (isinstance(value, float) and np.isnan(value)):
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


def build_game_text(game_row):
    game_name = str(game_row.get("game_name", ""))
    short_description = str(game_row.get("short_description", ""))

    genre_names = _to_text_list(game_row.get("genre_names"))
    category_names = _to_text_list(game_row.get("category_names"))

    joined_genres = ", ".join(genre_names)
    joined_categories = ", ".join(category_names)

    return " | ".join(
        [
            f"Title: {game_name}",
            f"Genres: {joined_genres}",
            f"Categories: {joined_categories}",
            f"Description: {short_description}",
        ]
    )


def build_numeric_feature_matrix(features_dataframe):
    selected_numeric_columns = [
        "price_gbp",
        "positive_review_ratio",
        "current_players",
        "market_player_review_ratio",
        "high_playtime_sentiment",
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
    ]

    numeric_dataframe = features_dataframe[selected_numeric_columns].copy()
    numeric_dataframe = numeric_dataframe.fillna(0)

    scaler = StandardScaler()
    scaled_numeric_features = scaler.fit_transform(numeric_dataframe)

    return scaled_numeric_features


def encode_text_embeddings(model, game_text_list):
    with torch.inference_mode():
        text_embeddings = model.encode(
            game_text_list,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    return text_embeddings


def _download_image(image_url):
    if not isinstance(image_url, str) or image_url.strip() == "":
        return None

    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def build_image_embedding_list(features_dataframe, image_model, images_per_game):
    image_embedding_list = []
    image_embedding_dimension = image_model.get_embedding_dimension()

    for _, game_row in features_dataframe.iterrows():
        screenshot_urls = _to_text_list(game_row.get("screenshot_urls"))
        header_image_url = game_row.get("header_image_url")

        selected_urls = screenshot_urls[:images_per_game]
        if len(selected_urls) == 0 and isinstance(header_image_url, str) and header_image_url.strip() != "":
            selected_urls = [header_image_url]

        image_batch = []
        for image_url in selected_urls:
            downloaded_image = _download_image(image_url)
            if downloaded_image is not None:
                image_batch.append(downloaded_image)

        if len(image_batch) == 0:
            image_embedding_list.append(np.zeros(image_embedding_dimension, dtype=np.float32))
            continue

        with torch.inference_mode():
            image_embeddings = image_model.encode(
                image_batch,
                batch_size=min(8, len(image_batch)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        image_embedding_list.append(np.mean(image_embeddings, axis=0).astype(np.float32))

    return np.vstack(image_embedding_list)


def build_combined_embeddings(text_embeddings, image_embeddings, numeric_matrix):
    text_weight = 0.55
    image_weight = 0.25
    numeric_weight = 0.20

    combined_embeddings = np.concatenate(
        [
            text_embeddings * text_weight,
            image_embeddings * image_weight,
            numeric_matrix * numeric_weight,
        ],
        axis=1,
    )

    combined_norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
    combined_norms[combined_norms == 0] = 1.0

    return combined_embeddings / combined_norms


def build_similarity_rows(features_dataframe, combined_embeddings, similar_games_count):
    normalized_embeddings = combined_embeddings.astype(np.float32)
    faiss.normalize_L2(normalized_embeddings)

    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)

    neighbor_count = min(similar_games_count + 1, len(features_dataframe))
    similarities_matrix, indices_matrix = index.search(normalized_embeddings, neighbor_count)

    similarity_rows = []

    for anchor_index in range(len(features_dataframe)):
        anchor_app_id = int(features_dataframe.iloc[anchor_index]["app_id"])
        anchor_game_name = features_dataframe.iloc[anchor_index]["game_name"]

        for neighbor_position in range(1, len(indices_matrix[anchor_index])):
            neighbor_index = indices_matrix[anchor_index][neighbor_position]
            neighbor_similarity = similarities_matrix[anchor_index][neighbor_position]

            neighbor_app_id = int(features_dataframe.iloc[neighbor_index]["app_id"])
            neighbor_game_name = features_dataframe.iloc[neighbor_index]["game_name"]

            similarity_rows.append(
                {
                    "anchor_app_id": anchor_app_id,
                    "anchor_game_name": anchor_game_name,
                    "neighbor_rank": neighbor_position,
                    "neighbor_app_id": neighbor_app_id,
                    "neighbor_game_name": neighbor_game_name,
                    "cosine_similarity": float(neighbor_similarity),
                }
            )

    return pd.DataFrame(similarity_rows), index


def save_recommendation_artifacts(
    artifacts_directory,
    features_dataframe,
    text_embeddings,
    image_embeddings,
    numeric_feature_matrix,
    combined_embeddings,
    similarity_dataframe,
    faiss_index,
):
    output_directory = Path(artifacts_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    embedding_matrix_path = output_directory / "game_genome_embeddings.npy"
    text_embeddings_path = output_directory / "game_text_embeddings.npy"
    image_embeddings_path = output_directory / "game_image_embeddings.npy"
    numeric_features_path = output_directory / "game_numeric_features.npy"
    metadata_path = output_directory / "game_genome_metadata.parquet"
    similarity_path = output_directory / "game_neighbors.parquet"
    similarity_csv_path = output_directory / "game_neighbors.csv"
    faiss_index_path = output_directory / "game_genome_faiss.index"

    np.save(embedding_matrix_path, combined_embeddings)
    np.save(text_embeddings_path, text_embeddings)
    np.save(image_embeddings_path, image_embeddings)
    np.save(numeric_features_path, numeric_feature_matrix)
    features_dataframe[
        [
            "app_id",
            "game_name",
            "primary_genre",
            "price_gbp",
            "positive_review_ratio",
            "header_image_url",
            "developer_names",
            "publisher_names",
            "genre_names",
            "category_names",
        ]
    ].to_parquet(
        metadata_path,
        index=False,
    )
    similarity_dataframe.to_parquet(similarity_path, index=False)
    similarity_dataframe.to_csv(similarity_csv_path, index=False)
    faiss.write_index(faiss_index, str(faiss_index_path))


def main():
    args = parse_args()

    model_device = "cuda" if torch.cuda.is_available() else "cpu"

    features_dataframe = pd.read_parquet(args.features_path)
    features_dataframe = features_dataframe.dropna(subset=["app_id", "game_name"]).copy()
    features_dataframe = features_dataframe.drop_duplicates(subset=["app_id"]).reset_index(drop=True)

    game_text_list = []
    for _, game_row in features_dataframe.iterrows():
        game_text_list.append(build_game_text(game_row))

    embedding_model = SentenceTransformer(args.embedding_model_name, device=model_device)
    image_embedding_model = SentenceTransformer(args.image_embedding_model_name, device=model_device)
    text_embeddings = encode_text_embeddings(embedding_model, game_text_list)
    image_embeddings = build_image_embedding_list(features_dataframe, image_embedding_model, args.images_per_game)
    numeric_feature_matrix = build_numeric_feature_matrix(features_dataframe)
    combined_embeddings = build_combined_embeddings(text_embeddings, image_embeddings, numeric_feature_matrix)

    similarity_dataframe, faiss_index = build_similarity_rows(
        features_dataframe=features_dataframe,
        combined_embeddings=combined_embeddings,
        similar_games_count=args.similar_games_count,
    )

    save_recommendation_artifacts(
        artifacts_directory=args.artifacts_directory,
        features_dataframe=features_dataframe,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        numeric_feature_matrix=numeric_feature_matrix,
        combined_embeddings=combined_embeddings,
        similarity_dataframe=similarity_dataframe,
        faiss_index=faiss_index,
    )


if __name__ == "__main__":
    main()
