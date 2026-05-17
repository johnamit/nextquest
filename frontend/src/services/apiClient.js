const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function requestJson(path, queryParams = {}) {
  const params = new URLSearchParams(queryParams);
  const queryString = params.toString();
  const url = queryString ? `${API_BASE_URL}${path}?${queryString}` : `${API_BASE_URL}${path}`;

  const response = await fetch(url);
  const data = await response.json();

  if (!response.ok) {
    const message = data?.message || "Request failed.";
    throw new Error(message);
  }

  return data;
}

async function getSteamLibrary(steamId) {
  return requestJson("/steam/library", { steam_id: steamId });
}

async function getGamesCatalog() {
  return requestJson("/games");
}

async function getRecommendations(params) {
  return requestJson("/recommendations", params);
}

async function getGameProfile(appId) {
  return requestJson("/game-profile", { app_id: String(appId) });
}

export {
  getSteamLibrary,
  getGamesCatalog,
  getRecommendations,
  getGameProfile,
};
