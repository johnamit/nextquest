import { useMemo } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAppSession } from "../context/AppSessionContext";
import GameProfileModal from "../components/GameProfileModal";
import nextQuestLogo from "../assets/nextquest.png";

function RecommendationsPage() {
  const navigate = useNavigate();

  const { recommendations, profileAppId, setProfileAppId } = useAppSession();

  const topRecommendations = useMemo(() => {
    return (recommendations || []).slice(0, 9);
  }, [recommendations]);

  // if no recommendation data in session, redirect to selection page
  if (!Array.isArray(recommendations) || recommendations.length === 0) {
    return <Navigate to="/select" replace />;
  }

  return (
    <main className="app">
      <section className="results-section">
        <img className="page-brand-logo" src={nextQuestLogo} alt="NextQuest" />
        <h1>Top Recommendations</h1>
        <p className="subtext">Based on your selected games</p>
  
        <div className="results-actions">
          <button className="manual-submit-btn" onClick={() => navigate("/select")}>
            Back to Game Selection
          </button>
        </div>
  
        <div className="manual-grid three-cols">
          {topRecommendations.map((game) => {
            const similarityPercent = Math.round((Number(game.cosine_similarity) || 0) * 100);
  
            return (
              <article
                key={game.app_id}
                className="manual-card recommendation-clickable"
                title={game.game_name}
                aria-label={game.game_name}
                style={{ backgroundImage: `url(${game.header_image_url || ""})` }}
                onClick={() => setProfileAppId(game.app_id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setProfileAppId(game.app_id);
                  }
                }}
              >
                <span className="match-pill">{similarityPercent}% match</span>
              </article>
            );
          })}
        </div>
      </section>
  
      <GameProfileModal
        appId={profileAppId}
        isOpen={profileAppId !== null}
        onClose={() => setProfileAppId(null)}
      />
    </main>
  );
}

export default RecommendationsPage;
