import React, { useState } from "react";
import ReactDOM from "react-dom/client";

import TennisCourt from "./TennisCourt.jsx";
import VideoUpload from "./VideoUpload.jsx";
import ShotChart from "./ShotChart.jsx";
import Header from "./header.jsx";

import {
  BrowserRouter,
  Routes,
  Route
} from "react-router-dom";

function App() {

  const [bounces, setBounces] = useState([]);

  function addBounce(x, y) {

    // creates a new array with the currentBounces (...currentBounces) and the new coordinates.
    // React prefers creating a new array because that gives a clear indication that the state changed
    setBounces((currentBounces) => [
      ...currentBounces,
      { x, y }             
    ])
  }

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

