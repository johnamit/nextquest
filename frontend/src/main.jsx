import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AppSessionProvider } from "./context/AppSessionContext";

import "./index.css";
import "./styles/tokens.css";
import "./styles/layout.css";
import "./styles/components.css";
import "./styles/pages/login.css";
import "./styles/pages/selection.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppSessionProvider>
        <App />
      </AppSessionProvider>
    </BrowserRouter>
  </React.StrictMode>
);
