import { createContext, useContext, useMemo, useState } from "react";

const AppSessionContext = createContext(null);

function AppSessionProvider({ children }) {
    const [steamId, setSteamId] = useState("");
    const [ownedGames, setOwnedGames] = useState([]);
    const [catalogGames, setCatalogGames] = useState([]);
    
    const [sourceMode, setSourceMode] = useState("steam_library"); // "steam_library" | "all_games"
    const [selectedAppIds, setSelectedAppIds] = useState([]);
    
    const [recommendations, setRecommendations] = useState([]);
    const [blockedAppIds, setBlockedAppIds] = useState([]);
    const [activeQuery, setActiveQuery] = useState(null);
    
    const [profileAppId, setProfileAppId] = useState(null);

    function resetSelectionState() {
        setSelectedAppIds([]);
        setSourceMode("steam_library");
    }

    function resetRecommendationState() {
        setRecommendations([]);
        setBlockedAppIds([]);
        setActiveQuery(null);
        setProfileAppId(null);
    }

    function resetAllSession() {
        setSteamId("");
        setOwnedGames([]);
        setCatalogGames([]);
        resetSelectionState();
        resetRecommendationState();
    }

    const value = useMemo(() => ({
        steamId,
        setSteamId,
        ownedGames,
        setOwnedGames,
        catalogGames,
        setCatalogGames,
        sourceMode,
        setSourceMode,
        selectedAppIds,
        setSelectedAppIds,
        recommendations,
        setRecommendations,
        blockedAppIds,
        setBlockedAppIds,
        activeQuery,
        setActiveQuery,
        profileAppId,
        setProfileAppId,
        resetSelectionState,
        resetRecommendationState,
        resetAllSession,
    }),
    [
        steamId,
        ownedGames,
        catalogGames,
        sourceMode,
        selectedAppIds,
        recommendations,
        blockedAppIds,
        activeQuery,
        profileAppId
    ]
    );
    return <AppSessionContext.Provider value={value}>{children}</AppSessionContext.Provider>;
}

function useAppSession() {
  const context = useContext(AppSessionContext);
  if (!context) {
    throw new Error("useAppSession must be used inside AppSessionProvider");
  }
  return context;
}
export { AppSessionProvider, useAppSession };