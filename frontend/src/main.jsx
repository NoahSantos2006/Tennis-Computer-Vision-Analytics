import React, { useState } from "react";
import ReactDOM from "react-dom/client";

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
    <BrowserRouter>

      <Header />

      <Routes>

        <Route
          path="/"
          element={
            <div>
              <VideoUpload />
            </div>
          }
        />

        <Route
          path="/results"
          element={
            <div>
              <ShotChart />
            </div>
          }
        />

      </Routes>
    
    </BrowserRouter>
  );

  

}

// find html element name 'root' and turn it into a React-controlled area and display App component inside it

ReactDOM.createRoot(
  document.getElementById("root")
  ).render(
  <App />
);

