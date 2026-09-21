import React, { useEffect, useRef, useState } from "react";
import "./ShotChart.css";
import TennisCourt from "./TennisCourt";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

function ShotChart() {

    useEffect(() => {

        // gets <video> element
        const video = videoRef.current

        if (!video) return

        let callbackId

        // loop through video
        const updateFrame = (now, metadata) => {

            // mediaTime return playback position in seconds
            const frame = Math.floor(metadata.mediaTime * fps)

            // sets the current frame
            setCurrentFrame(frame)

            // call updateFrame() on the next frame of the video
            callbackId = video.requestVideoFrameCallback(updateFrame)
        }

        // first request
        callbackId = video.requestVideoFrameCallback(updateFrame)

        return () => {
            video.cancelVideoFrameCallback(callbackId)
        }
    })

    const navigate = useNavigate()
    const location = useLocation()

    const data = location.state?.data || []

    const bounces_dict = data['Bounce Detection Dictionary']
    const fps = data['fps']
    const video_filename = data['video filename']
    const job_id = data['job id']    
    const videoRef = useRef(null)
    // currentFrame = current value
    // setCurrentFrame = function used to change the value
    // 0 = starting value
    const [currentFrame, setCurrentFrame] = useState(0)
    const [currentFilter, setFilter] = useState(0)

    let first_frame_bounce = Infinity
    let last_frame_bounce = -Infinity
    Object.keys(bounces_dict).map((key) => {

        const number = Number(key)

        if (number < first_frame_bounce) {
            first_frame_bounce = number
        }

        if (number > last_frame_bounce) {
            last_frame_bounce = number
        }

    })

    const rally_duration = (last_frame_bounce - first_frame_bounce) / fps

    let minute_duration = Math.floor(rally_duration / 60)
    let second_duration = Math.floor(rally_duration % 60)

    const total_shots = Object.keys(bounces_dict).length

    return (
        <div className="shot-chart-page">

            <main className="shot-chart-main">

                <div className="shot-chart-top-row">

                    <div>
                        <div className="shot-chart-section-label">
                            SHOT CHART
                        </div>

                        <h2>{video_filename}</h2>
                    </div>

                    <button 
                        className="new-video-back-button"
                        onClick={() => navigate("/")}
                    >
                        ← New video
                    </button>

                </div>

                <div className="shot-chart-stats">

                    <div className="shot-chart-stat-card">
                        <div className="shot-chart-stat-label">
                            TOTAL SHOTS
                        </div>

                        <div className="shot-chart-stat-number">
                            {total_shots}
                        </div>
                    </div>

                    <div className="shot-chart-stat-card">
                        <div className="shot-chart-stat-label">
                            RALLY DURATION
                        </div>

                        <div className="shot-chart-stat-number">
                            {minute_duration}m {second_duration}sec
                        </div>
                    </div>

                </div>

                <div className="shot-chart-content">

                    <aside>

                        <div className="shot-chart-filter-section">

                            <div className="shot-chart-filter-title">
                                FILTER BY SHOT
                            </div>

                            <button className={`shot-chart-filter-button ${currentFilter === 0 ? "active" : ""}`} onClick={() => setFilter(0)}>
                                All Shots
                            </button>

                            <button className={`shot-chart-filter-button ${currentFilter === 1 ? "active" : ""}`} onClick={() => setFilter(1)}>
                                Bounce
                            </button>

                            <button className={`shot-chart-filter-button ${currentFilter === 2 ? "active" : ""}`} onClick={() => setFilter(2)}>
                                Hit
                            </button>

                        </div>

                    </aside>

                    <section className="shot-chart-container">

                        <div className="shot-chart-chart-header">
                            <span>BIRD'S-EYE COURT VIEW</span>
                            <span>{total_shots} shots plotted</span>
                        </div>

                        <div className="shot-chart-court-area">
                            <TennisCourt 
                                bounces_dict={bounces_dict}
                                filter={currentFilter}
                            />
                        </div>

                    </section>

                </div>

                <section className="rally-playback-container">

                    <div className="rally-playback-header">
                        <span>RALLY PLAYBACK</span>
                        <span>FRAME {currentFrame}</span>
                    </div>

                    <div className="rally-playback-content">

                        <div className="rally-video-container">
                            <video
                                ref={videoRef}
                                className="rally-video"
                                controls
                            >
                                <source
                                    src={`http://localhost:8000/video/${job_id}/${video_filename}`}
                                    type="video/mp4"
                                />
                            </video>
                        </div>

                        <div className="rally-live-court">

                            <div className="rally-court-label">
                                LIVE COURT
                            </div>

                            <TennisCourt
                                bounces_dict={bounces_dict}
                                currentFrame={currentFrame}
                                playbackMode={true}
                            />

                        </div>

                    </div>

                </section>

            </main>

        </div>
    );
}

export default ShotChart;