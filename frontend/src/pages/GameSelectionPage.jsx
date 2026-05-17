import { useMemo, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAppSession } from "../context/AppSessionContext";
import { getGamesCatalog, getRecommendations } from "../services/apiClient";
import GameProfileModal from "../components/GameProfileModal";
import nextQuestLogo from "../assets/nextquest.png";

function GameSelectionPage() {
  const navigate = useNavigate();

  const {
    steamId,
    ownedGames,
    catalogGames,
    setCatalogGames,
    sourceMode,
    setSourceMode,
    selectedAppIds,
    setSelectedAppIds,
    setRecommendations,
    setBlockedAppIds,
    setActiveQuery,
    resetRecommendationState,
    resetAllSession,
  } = useAppSession();

  const [searchText, setSearchText] = useState("");
  const [isLoadingGames, setIsLoadingGames] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorText, setErrorText] = useState("");
  const [selectionProfileAppId, setSelectionProfileAppId] = useState(null);

  const hasSteamSession = steamId.trim() !== "" && ownedGames.length > 0;

  if (!hasSteamSession) {
    return <Navigate to="/" replace />;
  }

  const displayedGames = sourceMode === "steam_library" ? ownedGames : catalogGames;

  const filteredGames = useMemo(() => {
    const q = searchText.trim().toLowerCase();
    if (q === "") return displayedGames;
    return displayedGames.filter((game) =>
      String(game.game_name || "").toLowerCase().includes(q)
    );
  }, [displayedGames, searchText]);

  const ownedAppIdSet = useMemo(() => {
    return new Set((ownedGames || []).map((game) => Number(game.app_id)));
  }, [ownedGames]);

  async function handleSourceChange(nextMode) {
    if (nextMode === sourceMode) return;
    
    setErrorText("");
    setSourceMode(nextMode);
    
    // Reset selected games whenever source mode changes
    setSelectedAppIds([]);
    setSearchText("");
    
    if (nextMode === "all_games" && catalogGames.length === 0) {
      setIsLoadingGames(true);
      try {
        const result = await getGamesCatalog();
        setCatalogGames(result.data || []);
      } catch (error) {
        setErrorText(error.message || "Could not load catalog games.");
      } finally {
        setIsLoadingGames(false);
      }
    }
  }

  function toggleSelectedApp(appId) {
    setSelectedAppIds((prev) => {
      if (prev.includes(appId)) {
        return prev.filter((id) => id !== appId);
      }
      return [...prev, appId];
    });
  }

  async function handleGetRecommendations() {
    setErrorText("");

    if (selectedAppIds.length === 0) {
      setErrorText("Select at least one game.");
      return;
    }
    
    setIsSubmitting(true);
    try {
      const likedAppIdsText = selectedAppIds.join(",");
      const result = await getRecommendations({
        liked_app_ids: likedAppIdsText,
        top_k: "9",
        steam_id: steamId,
        exclude_owned: "true",
      });
    
      const rows = result.data || [];
      if (rows.length === 0) {
        throw new Error(result.message || "No recommendations found.");
      }
     
      resetRecommendationState();
      setBlockedAppIds([]);
      setRecommendations(rows);
      setActiveQuery({
        mode: "manual",
        liked_app_ids: likedAppIdsText,
      });
      
      navigate("/recommendations");
    } catch (error) {
      setErrorText(error.message || "Could not generate recommendations.");
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleLogout() {
    resetAllSession();
    navigate("/");
  }

  return (
    <main className="app">
      <section className="results-section">
        <div className="page-topbar">
          <button type="button" className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>
        <img className="page-brand-logo" src={nextQuestLogo} alt="NextQuest" />
        <p className="subtext">Select games you like from your Steam library or from the catalog of games.</p>
      
        <div className="mode-switch">
          <button
            className={sourceMode === "steam_library" ? "mode-btn active" : "mode-btn"}
            onClick={() => handleSourceChange("steam_library")}
            disabled={isLoadingGames || isSubmitting}
          >
            Steam Library
          </button>
      
          <button
            className={sourceMode === "all_games" ? "mode-btn active" : "mode-btn"}
            onClick={() => handleSourceChange("all_games")}
            disabled={isLoadingGames || isSubmitting}
          >
            All Games
          </button>
        </div>
      
        <input
          type="text"
          className="search-input"
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          placeholder="Search games..."
          disabled={isLoadingGames || isSubmitting}
        />

        <div className="results-actions">
          <p className="subtext selection-count">
            Selected: {selectedAppIds.length} {selectedAppIds.length === 1 ? "game" : "games"}
          </p>
          <button
            className="manual-submit-btn"
            onClick={handleGetRecommendations}
            disabled={isSubmitting || isLoadingGames || selectedAppIds.length === 0}
          >
            {isSubmitting ? "Loading..." : "Get Recommendations"}
          </button>
        </div>
       
        {isLoadingGames && <p className="subtext">Loading games...</p>}
      
        {!isLoadingGames && (
          <div className="game-grid-shell">
            <div className="manual-grid three-cols">
              {filteredGames.map((game) => {
                const appId = Number(game.app_id);
                const isChecked = selectedAppIds.includes(appId);
                const inLibrary = ownedAppIdSet.has(appId);

                return (
                  <label
                    key={appId}
                    className={isChecked ? "manual-card checked" : "manual-card"}
                    title={game.game_name}
                    aria-label={game.game_name}
                    style={{ backgroundImage: `url(${game.header_image_url || ""})` }}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => toggleSelectedApp(appId)}
                      disabled={isSubmitting}
                    />
                    {inLibrary && <span className="library-pill">In Library</span>}
                    <button
                      type="button"
                      className="card-info-btn"
                      aria-label={`View details for ${game.game_name}`}
                      onClick={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        setSelectionProfileAppId(appId);
                      }}
                    >
                      <span className="material-symbols-outlined card-info-icon">info</span>
                    </button>
                  </label>
                );
              })}
            </div>
          </div>
        )}

        {errorText !== "" && <p className="error-text">{errorText}</p>}
      </section>

      <GameProfileModal
        appId={selectionProfileAppId}
        isOpen={selectionProfileAppId !== null}
        onClose={() => setSelectionProfileAppId(null)}
      />
    </main>
  );
}

export default GameSelectionPage;
