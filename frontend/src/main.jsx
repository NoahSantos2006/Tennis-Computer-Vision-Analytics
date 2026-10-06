import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import { Analytics } from "@vercel/analytics/react"

import ServerWakeup from "./ServerWakeup.jsx"
import VideoUpload from "./VideoUpload.jsx";
import ShotChart from "./ShotChart.jsx";
import Header from "./header.jsx";

import {
  BrowserRouter,
  Routes,
  Route
} from "react-router-dom";

function App() {
  return (
    <>
      <ServerWakeup apiUrl={import.meta.env.VITE_API_URL}>
        <BrowserRouter>
          <Header />
          <Routes>
            <Route path="/" element={<VideoUpload />} />
            <Route path="/results" element={<ShotChart />} />
          </Routes>
        </BrowserRouter>
      </ServerWakeup>
    <Analytics />
    </>
  );
}

// find html element name 'root' and turn it into a React-controlled area and display App component inside it

ReactDOM.createRoot(
  document.getElementById("root")
  ).render(
  <App />
);

