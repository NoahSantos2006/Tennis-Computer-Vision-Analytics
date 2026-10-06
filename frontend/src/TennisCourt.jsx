import React from "react";

import tennis_ball from "./images/tennis_ball.svg"
import tennis_racket from "./images/tennis_racket.svg"
import "./TennisCourt.css"

function TennisCourt({ 
  bounces_dict = {},
  currentFrame = null,
  playbackMode = false,
  filter = 0,
}) {

  const COURT_LENGTH = 23.77;
  const COURT_WIDTH = 18;               // original 10.97
  const DOUBLES_ALLEY = 2.25;              // original 1.37
  const SERVICE_LINE_FROM_NET = 6.4;
  const MIN_MARGIN = 3;                   // minimum space around the court
  const BALL_SIZE = 0.5;
  const RACKET_SIZE = 1.3;

  function homographyToSvg(x, y) {
    const HOMOGRAPHY_PADDING = 30;
    const HOMOGRAPHY_SCALE = 25;
    const REAL_COURT_WIDTH = 10.97;

    // Homography pixels -> real court coordinates
    const courtX = (x - HOMOGRAPHY_PADDING) / HOMOGRAPHY_SCALE;
    const courtY = (y - HOMOGRAPHY_PADDING) / HOMOGRAPHY_SCALE;

    // Real court coordinates -> your wider SVG court
    return {
      x: courtX * (COURT_WIDTH / REAL_COURT_WIDTH),
      y: courtY
    };
  }

  let scaled_bounces = Object.fromEntries(
    Object.entries(bounces_dict)
      .filter(([, values]) => values["homography location"])
      .map(([frame_id, values]) => {

    let location = values["homography location"]
    const scaled_location = homographyToSvg(location[0], location[1])
    let label = values['label']
    
    return [
        frame_id, 
        {
          ...scaled_location,
          label: label
        }
      ]
    })
  )

  // Grow each side's margin so every marker (even hits far behind the baseline) stays in view.
  // Uses all events, not just the ones shown so far, so the live court doesn't resize during playback.
  let marginTop = MIN_MARGIN
  let marginBottom = MIN_MARGIN
  let marginLeft = MIN_MARGIN
  let marginRight = MIN_MARGIN

  Object.values(scaled_bounces).forEach(({ x, y, label }) => {
    const reach = (label === 2 ? RACKET_SIZE : BALL_SIZE) / 2 + 0.5

    marginTop = Math.max(marginTop, reach - y)
    marginBottom = Math.max(marginBottom, y + reach - COURT_LENGTH)
    marginLeft = Math.max(marginLeft, reach - x)
    marginRight = Math.max(marginRight, x + reach - COURT_WIDTH)
  })

  // Keep the court centered
  const marginY = Math.max(marginTop, marginBottom)
  const marginX = Math.max(marginLeft, marginRight)

  // In playback, the most recent shot that has happened so far gets highlighted
  let latestFrame = null
  if (playbackMode) {
    Object.keys(scaled_bounces).forEach((frame_id) => {
      const frameNumber = Number(frame_id)
      if (frameNumber <= currentFrame && (latestFrame === null || frameNumber > latestFrame)) {
        latestFrame = frameNumber
      }
    })
  }

  const HALF_LENGTH = COURT_LENGTH / 2;
  const HALF_WIDTH = COURT_WIDTH / 2;

  const topServiceY = HALF_LENGTH - SERVICE_LINE_FROM_NET;
  const bottomServiceY = HALF_LENGTH + SERVICE_LINE_FROM_NET;

  // CourtVision color palette
  const COURT_COLOR = "#D8DFC5";
  const COURT_LINE = "#A8B58C";
  const NET_COLOR = "#4A5E35";
  const RUNOFF_COLOR = "#E8ECDC";

  return (
    <svg
      viewBox={`${-marginX} ${-marginY} ${COURT_WIDTH + 2 * marginX} ${COURT_LENGTH + 2 * marginY}`}
      overflow="visible"
    >

      {/* Run-off area */}
      <rect
        x={-marginX}
        y={-marginY}
        width={COURT_WIDTH + 2 * marginX}
        height={COURT_LENGTH + 2 * marginY}
        fill={RUNOFF_COLOR}
      />

      {/* Court */}
      <rect
        x="0"
        y="0"
        width={COURT_WIDTH}
        height={COURT_LENGTH}
        fill={COURT_COLOR}
      />

      {/* Outer court lines */}
      <rect
        x="0"
        y="0"
        width={COURT_WIDTH}
        height={COURT_LENGTH}
        fill="none"
        stroke={COURT_LINE}
        strokeWidth=".08"
      />

      {/* Singles sidelines */}
      <line
        x1={DOUBLES_ALLEY}
        y1="0"
        x2={DOUBLES_ALLEY}
        y2={COURT_LENGTH}
        stroke={COURT_LINE}
        strokeWidth=".08"
      />

      <line
        x1={COURT_WIDTH - DOUBLES_ALLEY}
        y1="0"
        x2={COURT_WIDTH - DOUBLES_ALLEY}
        y2={COURT_LENGTH}
        stroke={COURT_LINE}
        strokeWidth=".08"
      />

      {/* Service lines */}
      <line
        x1={DOUBLES_ALLEY}
        y1={topServiceY}
        x2={COURT_WIDTH - DOUBLES_ALLEY}
        y2={topServiceY}
        stroke={COURT_LINE}
        strokeWidth=".08"
      />

      <line
        x1={DOUBLES_ALLEY}
        y1={bottomServiceY}
        x2={COURT_WIDTH - DOUBLES_ALLEY}
        y2={bottomServiceY}
        stroke={COURT_LINE}
        strokeWidth=".08"
      />

      {/* Center service line */}
      <line
        x1={HALF_WIDTH}
        y1={topServiceY}
        x2={HALF_WIDTH}
        y2={bottomServiceY}
        stroke={COURT_LINE}
        strokeWidth=".08"
      />

      {/* Net */}
      <line
        x1="0"
        y1={HALF_LENGTH}
        x2={COURT_WIDTH}
        y2={HALF_LENGTH}
        stroke={NET_COLOR}
        strokeWidth=".12"
      />

      {/* Bounces */}
      {Object.entries(scaled_bounces).map(([frame_id, values]) => {
        // bounce

        const frameNumber = Number(frame_id)

        if (playbackMode && frameNumber > currentFrame) {
          return null
        }

        let current_href
        let size

        if (values['label'] === 1) {
          current_href = tennis_ball
          size = BALL_SIZE
        } else if (values['label'] === 2) {
          current_href = tennis_racket
          size = RACKET_SIZE
        }
        
        
        const isLatest = playbackMode && frameNumber === latestFrame

        let markerClass = "shot-marker"
        if (isLatest) {
          markerClass += " shot-marker-latest"
        } else if (playbackMode) {
          markerClass += " shot-marker-past"
        }

        if (filter == 0 || filter == values['label']) {
         return (
          <g key={`shot-marker-frame-${frame_id}`}>
            {/* Ripple ring on the newest shot (keyed by frame so it replays for each new shot) */}
            {isLatest && (
              <circle
                key={`shot-ripple-${frame_id}`}
                className="shot-ripple"
                cx={values["x"]}
                cy={values["y"]}
                r={size}
                fill="none"
                stroke={NET_COLOR}
                strokeWidth=".1"
              />
            )}

            <image
              className={markerClass}
              href={current_href}
              x={values["x"] - size / 2}
              y={values["y"] - size / 2}
              width={size}
              height={size}
              color="#5F7F68"
            />
          </g>
         )
        }
      })}

    </svg>
  );
}

export default TennisCourt;