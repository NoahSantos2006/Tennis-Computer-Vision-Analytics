import React from "react";
import "./ProcessingVideo.css";

function ProcessingVideo({ status = null, progress = 0, currentFrame = 0, totalFrames = 0, stage = null}) {

    // 0 = not started
    // 1 = in progress
    // 2 = finished
    let videoValidation = 0;
    let videoProcessing = 0;
    let videoXGBoostPrediction = 0;

    if (status === "finished") {
        videoValidation = 2;
        videoProcessing = 2;
        videoXGBoostPrediction = 2;
    } else {
        switch (stage) {
            case "Validating Video":
                videoValidation = 1;
                break;

            case "Processing Frames":
                videoValidation = 2;
                videoProcessing = 1;
                break;

            case "Detecting Bounces and Hits":
                videoValidation = 2;
                videoProcessing = 2;
                videoXGBoostPrediction = 1;
                break;
        }
    }


    return (
        <div className="processing-page">

            <div className="processing-card">

                <div className="processing-label">
                    VIDEO ANALYSIS
                </div>

                <h1>Analyzing your video</h1>

                <p className="processing-description">
                    Detecting players, ball movement, and court position.
                </p>

                <div className="processing-progress-info">
                    <span>{stage}</span>
                    <span>{Math.round(progress)}%</span>
                </div>

                <div className="processing-progress-track">
                    <div
                        className="processing-progress-bar"
                        style={{ width: `${progress}%` }}
                    />
                </div>

                <div className="processing-frame-count">
                    Frame {currentFrame} of {totalFrames}
                </div>

                <div className="processing-status">

                    {
                        videoValidation === 0
                        ? <div className="waiting">
                            <span>○</span>
                            Validating Video
                        </div>
                        : videoValidation === 1
                            ? <div className="active">
                                <span className="processing-dot"></span>
                                Validating Video
                            </div>
                            : <div className="complete">
                                <span>✓</span>
                                Validating Video
                            </div>
                    }

                    {
                        videoProcessing === 0
                        ? <div className="waiting">
                            <span>○</span>
                            Processing Frames
                        </div>
                        : videoProcessing === 1
                            ? <div className="active">
                                <span className="processing-dot"></span>
                                Processing Frames
                            </div>
                            : <div className="complete">
                                <span>✓</span>
                                Processing Frames
                            </div>
                    }

                    {
                        videoXGBoostPrediction === 0
                        ? <div className="waiting">
                            <span>○</span>
                            Detecting Bounces and Hits
                        </div>
                        : videoXGBoostPrediction === 1
                            ? <div className="active">
                                <span className="processing-dot"></span>
                                Detecting Bounces and Hits
                            </div>
                            : <div className="complete">
                                <span>✓</span>
                                Detecting Bounces and Hits
                            </div>
                    }

                </div>

                <p className="processing-note">
                    This may take a few minutes.
                </p>

            </div>

        </div>
    );
}

export default ProcessingVideo;