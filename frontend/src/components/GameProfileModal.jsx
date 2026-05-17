import { useEffect, useState } from "react";
import { getGameProfile } from "../services/apiClient";

function formatPrice(price) {
  if (price === null || price === undefined || Number.isNaN(Number(price))) return "N/A";
  return `GBP ${Number(price).toFixed(2)}`;
}

function formatPlayers(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toLocaleString();
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function normalizeTagValue(raw) {
  if (raw === null || raw === undefined) return "";

  let text = String(raw).trim();
  text = text.replace(/^\[/, "").replace(/\]$/, "").trim();
  text = text.replace(/^['"]+|['"]+$/g, "").trim();
  return text;
}

function normalizeTagList(values) {
  if (!Array.isArray(values)) return [];

  const expanded = [];

  for (const raw of values) {
    const text = normalizeTagValue(raw);
    if (text === "") continue;

    const quotedMatches = [...text.matchAll(/'([^']+)'|"([^"]+)"/g)];
    if (quotedMatches.length > 0) {
      quotedMatches.forEach((match) => {
        const token = (match[1] || match[2] || "").trim();
        if (token !== "") expanded.push(token);
      });
      continue;
    }

    if (text.includes(",")) {
      text
        .split(",")
        .map((part) => part.trim())
        .filter((part) => part !== "")
        .forEach((part) => expanded.push(part));
      continue;
    }

    expanded.push(text);
  }

  const seen = new Set();
  const unique = [];
  for (const item of expanded) {
    const key = item.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(item);
  }
  return unique;
}

function buildTagList(profile) {
  const tags = [];

  if (profile?.year) {
    tags.push(String(profile.year));
  }

  const studio = normalizeTagValue(profile?.studio);
  if (studio !== "") {
    tags.push(studio);
  }

  if (Array.isArray(profile?.main_genres) && profile.main_genres.length > 0) {
    const mainGenres = normalizeTagList(profile.main_genres);
    tags.push(...mainGenres.slice(0, 2));
  } else if (Array.isArray(profile?.genres)) {
    const fallbackGenres = normalizeTagList(profile.genres);
    tags.push(...fallbackGenres.slice(0, 2));
  }

  return tags.slice(0, 4);
}

function getScoreTierClass(scoreValue) {
  if (scoreValue === null || scoreValue === undefined || scoreValue === "") {
    return "score-tier-none";
  }

  const score = Number(scoreValue);
  if (Number.isNaN(score)) return "score-tier-none";
  if (score < 30) return "score-tier-very-low";
  if (score < 50) return "score-tier-low";
  if (score < 70) return "score-tier-mid";
  if (score < 90) return "score-tier-high";
  return "score-tier-top";
}

function getScreenshotUrls(profile) {
  if (!profile) return [];
  if (!Array.isArray(profile.screenshot_urls)) return [];

  const uniqueUrls = [];
  const seen = new Set();

  for (const rawUrl of profile.screenshot_urls) {
    const url = String(rawUrl || "").trim();
    if (url === "" || seen.has(url)) continue;
    seen.add(url);
    uniqueUrls.push(url);
    if (uniqueUrls.length >= 4) break;
  }

  return uniqueUrls;
}

function GameProfileModal({ appId, isOpen, onClose }) {
  const [profile, setProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorText, setErrorText] = useState("");

  useEffect(() => {
    if (!isOpen || !appId) return;
    
    let isMounted = true;

    async function loadProfile() {
      setIsLoading(true);
      setErrorText("");
      try {
        const data = await getGameProfile(appId);
        if (isMounted) setProfile(data);
      } catch (error) {
        if (isMounted) setErrorText(error.message || "Could not load game profile.");
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }
    
    loadProfile();
    
    return () => {
      isMounted = false;
    };
  }, [appId, isOpen]);
  
  if (!isOpen) return null;

  const screenshotUrls = getScreenshotUrls(profile);
  
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose} aria-label="Close profile">
          ×
        </button>
        {isLoading && <p className="subtext">Loading game profile...</p>}
        {errorText !== "" && <p className="error-text">{errorText}</p>}
        {!isLoading && !errorText && profile && (
          <>
            <div className="modal-header-row">
              <div>
                <h2 className="modal-title">{profile.game_name || "Unknown Game"}</h2>
                <div className="modal-tags">
                  {buildTagList(profile).map((tag, idx) => (
                    <span className="modal-tag-chip" key={`${tag}-${idx}`}>
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <div className={`modal-score-badge ${getScoreTierClass(profile.metacritic_score)}`}>
                {profile.metacritic_score ?? "N/A"}
              </div>
            </div>

            {profile.header_image_url ? (
              <img
                className="modal-cover"
                src={profile.header_image_url}
                alt={profile.game_name || "Game cover"}
              />
            ) : null}

            <div className="modal-metrics-grid">
              <div className="modal-metric-card">
                <span className="modal-metric-label">Current Price</span>
                <strong className="modal-metric-value">{formatPrice(profile.price_gbp)}</strong>
              </div>
              <div className="modal-metric-card">
                <span className="modal-metric-label">Current Players</span>
                <strong className="modal-metric-value">{formatPlayers(profile.current_players)}</strong>
              </div>
              <div className="modal-metric-card">
                <span className="modal-metric-label">Positive Review Ratio</span>
                <strong className="modal-metric-value">{formatPercent(profile.positive_review_ratio)}</strong>
              </div>
            </div>

            <div className="modal-reviews">
              <h3>Description</h3>
              {profile.description ? (
                <p className="modal-description">{profile.description}</p>
              ) : (
                <p className="subtext">No description available.</p>
              )}
            </div>

            {screenshotUrls.length > 0 && (
              <div className="modal-screenshots">
                <h3>Screenshots</h3>
                <div className="modal-screenshot-grid">
                  {screenshotUrls.map((url, index) => (
                    <img
                      key={`${url}-${index}`}
                      className="modal-screenshot"
                      src={url}
                      alt={`${profile.game_name || "Game"} screenshot ${index + 1}`}
                      loading="lazy"
                    />
                  ))}
                </div>
              </div>
            )}
           
          </>
        )}
      </div>
    </div>
  );
}

export default GameProfileModal;
