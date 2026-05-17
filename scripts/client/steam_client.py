import os
import time
import requests


class SteamClient:
    def __init__(self, request_timeout_seconds=30, request_delay_seconds=1.0):
        self.request_timeout_seconds = request_timeout_seconds
        self.request_delay_seconds = request_delay_seconds
        self.steam_api_key = os.getenv("STEAM_API_KEY")
        self.http_session = requests.Session()

    def _request_json(self, endpoint_url, request_params):
        retries_remaining = 3
        last_exception = None

        while retries_remaining > 0:
            try:
                response = self.http_session.get(
                    endpoint_url,
                    params=request_params,
                    timeout=self.request_timeout_seconds,
                )
                response.raise_for_status()
                time.sleep(self.request_delay_seconds)
                return response.json()
            except requests.RequestException as request_exception:
                last_exception = request_exception
                retries_remaining -= 1
                time.sleep(1.5)

        raise last_exception

    def get_app_details(self, app_id):
        endpoint_url = "https://store.steampowered.com/api/appdetails"
        request_params = {
            "appids": app_id,
            "cc": "GB",
            "l": "en",
        }
        response_json = self._request_json(endpoint_url, request_params)
        return response_json.get(str(app_id), {})

    def get_reviews(self, app_id, number_of_reviews=100):
        endpoint_url = f"https://store.steampowered.com/appreviews/{app_id}"
        request_params = {
            "json": 1,
            "language": "english",
            "num_per_page": number_of_reviews,
            "purchase_type": "all",
            "filter": "recent",
        }
        return self._request_json(endpoint_url, request_params)

    def get_current_players(self, app_id):
        endpoint_url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
        request_params = {
            "key": self.steam_api_key,
            "appid": app_id,
        }
        response_json = self._request_json(endpoint_url, request_params)
        return response_json.get("response", {}).get("player_count", 0)

    def get_owned_games(self, steam_id, include_appinfo=True, include_played_free_games=True):
        endpoint_url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
        request_params = {
            "key": self.steam_api_key,
            "steamid": steam_id,
            "include_appinfo": 1 if include_appinfo else 0,
            "include_played_free_games": 1 if include_played_free_games else 0,
        }

        response_json = self._request_json(endpoint_url, request_params)
        return response_json.get("response", {})
