import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from scripts.client.steam_client import SteamClient
from scripts.pipeline.recommend_from_games import generate_recommendations

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--artifacts-directory", default="data/artifacts")
    parser.add_argument("--features-path", default="data/features/game_features.parquet")
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--steam-id", default="76561198916832123")
    parser.add_argument("--liked-game-names", default="")
    parser.add_argument("--max-liked-games", type=int, default=20)
    
    return parser.parse_args()


# steam library mode: fetch and normalize owned games dataframe
def fetch_owned_games_dataframe(steam_client, steam_id):
    payload = steam_client.get_owned_games(steam_id)
    
    games = payload.get("games", [])
    if not games:
        return pd.DataFrame(columns=["app_id", "playtime_forever"])
    
    df = pd.DataFrame(games)
    df["app_id"] = pd.to_numeric(df.get("appid"), errors="coerce")
    df["playtime_forever"] = pd.to_numeric(df.get("playtime_forever"), errors="coerce").fillna(0)
    df = df.dropna(subset=["app_id"]).copy()
    df["app_id"] = df["app_id"].astype("int64")
    df = df.drop_duplicates(subset=["app_id"]).copy()
    
    return df


# keep only the games that the model knows
def load_known_app_ids(artifacts_directory):
    metadata_path = Path(artifacts_directory) / "game_genome_metadata.parquet"
    
    metadata_df = pd.read_parquet(metadata_path)
    metadata_df["app_id"] = pd.to_numeric(metadata_df["app_id"], errors="coerce")
    metadata_df = metadata_df.dropna(subset=["app_id"]).copy()
    metadata_df["app_id"] = metadata_df["app_id"].astype("int64")
    
    return set(metadata_df["app_id"].tolist())


def select_liked_app_ids(owned_df, known_app_ids, max_liked_games):
    matched_df = owned_df[owned_df["app_id"].isin(known_app_ids)].copy()
    matched_df = matched_df.sort_values(by="playtime_forever", ascending=False)
    
    liked_app_ids = matched_df["app_id"].head(max_liked_games).tolist()
    
    return liked_app_ids, matched_df


def main():
    load_dotenv()
    args = parse_args()

    # Mode 1: Manually liked games
    if args.liked_game_names.strip() != "":
        recommendations_df = generate_recommendations(
            artifacts_directory=args.artifacts_directory,
            features_path=args.features_path,
            liked_app_ids_text="",
            liked_game_names_text=args.liked_game_names,
            top_k=args.top_k,
        )

        if recommendations_df.empty:
            print("No recommendations generated.")
            return
        
        print("Mode: manual liked game names")
        print(recommendations_df[["game_name", "cosine_similarity", "why_recommended"]].to_string(index=False))
        return
    
    # Mode 2: Steam library
    steam_client = SteamClient()

    owned_df = fetch_owned_games_dataframe(steam_client, args.steam_id)
    if owned_df.empty:
        print("No owned games returned (private/unavailable profile, invalid steam id, or empty library).")
        return
    
    known_app_ids = load_known_app_ids(args.artifacts_directory)
    
    liked_app_ids, matched_df = select_liked_app_ids(
        owned_df=owned_df,
        known_app_ids=known_app_ids,
        max_liked_games=args.max_liked_games,
    )
    if not liked_app_ids:
        print("No overlap between Steam library and local model catalog.")
        return
    liked_app_ids_text = ",".join(str(app_id) for app_id in liked_app_ids)
    
    recommendations_df = generate_recommendations(
        artifacts_directory=args.artifacts_directory,
        features_path=args.features_path,
        liked_app_ids_text=liked_app_ids_text,
        liked_game_names_text="",
        top_k=args.top_k,
    )
    if recommendations_df.empty:
        print("No recommendations generated.")
        return
    
    print("Mode: Steam library liked games")
    print(f"Owned games: {len(owned_df)}")
    print(f"Matched to model catalog: {len(matched_df)}")
    print(f"Using liked app ids: {len(liked_app_ids)}")
    print(recommendations_df[["game_name", "cosine_similarity", "why_recommended"]].to_string(index=False))


if __name__ == "__main__":
    main()
