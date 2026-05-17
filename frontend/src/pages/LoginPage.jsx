import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAppSession } from "../context/AppSessionContext";
import { getSteamLibrary } from "../services/apiClient";
import nextQuestLogo from "../assets/nextquest.png";
const STEAM_ID_LENGTH = 17;

function sanitizeSteamId(value) {
  return String(value || "").replace(/\D/g, "").slice(0, STEAM_ID_LENGTH);
}

function isValidSteamId(value) {
  return /^\d{17}$/.test(value);
}

function LoginPage() {
  const navigate = useNavigate();

  const {
    steamId,
    setSteamId,
    setOwnedGames,
    resetSelectionState,
    resetRecommendationState,
  } = useAppSession();

  const [isLoading, setIsLoading] = useState(false);
  const [errorText, setErrorText] = useState("");
  const sanitizedSteamId = useMemo(() => sanitizeSteamId(steamId), [steamId]);
  const isComplete = sanitizedSteamId.length === STEAM_ID_LENGTH;

  async function handleLogin(event) {
    event.preventDefault();
    setErrorText("");
    const trimmedSteamId = sanitizeSteamId(steamId);
    if (trimmedSteamId === "") {
      setErrorText("Please enter your SteamID");
      return;
    }
    if (!isValidSteamId(trimmedSteamId)) {
      setErrorText("SteamID must be exactly 17 digits");
      return;
    }
    setIsLoading(true);
    try {
      const result = await getSteamLibrary(trimmedSteamId);
      if (!result.data || result.data.length === 0) {
        throw new Error(result.message || "Could not access Steam profile. Check SteamID and privacy settings.");
      }
      setSteamId(trimmedSteamId);
      setOwnedGames(result.data);
      resetSelectionState();
      resetRecommendationState();

      navigate("/select");
    } catch (error) {
      setErrorText(error.message || "Could not access Steam profile. Check SteamID and privacy settings.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app">
      <section className="login-v3-page">
        <img className="login-v3-logo" src={nextQuestLogo} alt="NextQuest" />
        <form onSubmit={handleLogin} className="login-v3-form">
          <div className="login-v3-input-row">
            <input
              type="text"
              value={sanitizedSteamId}
              onChange={(event) => setSteamId(sanitizeSteamId(event.target.value))}
              placeholder="Enter your 17 digit Steam ID"
              inputMode="numeric"
              autoComplete="off"
              maxLength={STEAM_ID_LENGTH}
              className="login-v3-input"
              aria-label="SteamID"
              disabled={isLoading}
            />
          </div>

          <div className="login-v3-arrow-row">
            <button
              type="submit"
              className={isComplete ? "login-v3-arrow-btn is-visible" : "login-v3-arrow-btn"}
              aria-label={isLoading ? "Connecting" : "Connect"}
              disabled={isLoading || !isComplete}
            >
            {isLoading ? ("...") : (<span className="material-symbols-outlined login-v3-arrow-icon" aria-hidden="true">arrow_forward_ios</span>)}
            </button>
          </div>
        </form>
        {errorText !== "" && <p className="error-text login-v3-error is-visible">{errorText}</p>}
      </section>
    </main>
  );
}

export default LoginPage;
