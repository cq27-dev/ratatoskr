import React from "react";
import { createRoot } from "react-dom/client";
import { NuqsAdapter } from "nuqs/adapters/react";
import App from "./App";
import "@xyflow/react/dist/style.css";
import "./style.css";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
  <React.StrictMode>
    {/* Plain React, no router: the adapter is what gives nuqs a history to write to. */}
    <NuqsAdapter>
      <App />
    </NuqsAdapter>
  </React.StrictMode>,
);
