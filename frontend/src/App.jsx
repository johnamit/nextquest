import { Navigate, Route, Routes } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import GameSelectionPage from "./pages/GameSelectionPage";
import RecommendationsPage from "./pages/RecommendationsPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LoginPage />} />
      <Route path="/select" element={<GameSelectionPage />} />
      <Route path="/recommendations" element={<RecommendationsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
