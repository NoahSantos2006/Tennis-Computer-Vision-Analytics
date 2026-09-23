import React, { useEffect, useState } from "react";
import "./VideoUpload.css";
import { Navigate, useNavigate } from "react-router-dom";
import ProcessingVideo from "./ProcessingVideo";

const API_URL = import.meta.env.VITE_API_URL;

function VideoUpload() {

    const navigate = useNavigate()

    const [jobId, setJobId] = useState(null)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [currentProgress, setCurrentProgress] = useState(0)
    const [currentFrame, setCurrentFrame] = useState(0)
    const [currentTotalFrames, setCurrentTotalFrames] = useState(0)
    const [currentStage, setCurrentStage] = useState(0)
    const [currentStatus, setCurrentStatus] = useState(0)

    useEffect(() => {

        if (!jobId) {
            return;
        }

        const interval = setInterval(async () => {

            const response = await fetch(
                `${API_URL}/jobs/${jobId}/status`
            )

            const data = await response.json();

            setCurrentProgress(data['progress'])
            setCurrentFrame(data['current frame'])
            setCurrentTotalFrames(data['total frames'])
            setCurrentStage(data["stage"])
            setCurrentStatus(data['status'])

            if (data['status'] === 'finished') {

                clearInterval(interval)

                const resultsResponse = await fetch(
                    `${API_URL}/jobs/${jobId}/results`
                )

                const resultsData = await resultsResponse.json()

                navigate("/results", {
                    state: {
                        data: resultsData
                    }
                })
            }

        }, 500)

        return () => {
            clearInterval(interval)
        }

    // dependency array (run this effect when the component first appears, and run it again if jobId or navigate changes)
    }, [jobId, navigate])


    // Sends a file to Python backend
    
    async function uploadVideo(file) {

        // Create an object that can hold files for an HTTP request
        const formData = new FormData();

        // Add the file under the name "video"
        formData.append("video", file);

        // Send HTTP POST request to Python server

        try {

            setIsAnalyzing(true)

            const response = await fetch(`${API_URL}/analyze`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Upload failed")
            }

            const data = await response.json()

            setJobId(data['job id'])

        } catch (error) {

            setIsAnalyzing(false)
            console.error("Error uploading video:", error)
        }

    }


    // Runs when user drops a file
    function handleDrop(event) {

        // Prevent browser from opening the dropped file
        event.preventDefault();

        // Get first dropped file
        const file = event.dataTransfer.files[0];

        if (file) {
            uploadVideo(file);
        }
    }


    // Allows files to be dragged over the dropzone
    function handleDragOver(event) {
        event.preventDefault();
    }


    // Runs when user chooses a file through Browse
    function handleFileSelect(event) {

        const file = event.target.files[0];

        if (file) {
            uploadVideo(file);
        }
    }

    return (
        <section className="video-upload-page">


            {isAnalyzing ? (
                <ProcessingVideo 
                    status={currentStatus}
                    progress={currentProgress}
                    currentFrame={currentFrame}
                    totalFrames={currentTotalFrames}
                    stage={currentStage}
                />
            ) : (

                <div>
                    <div className="video-upload-step">
                        UPLOAD RALLY VIDEO
                    </div>

                    <div
                        className="video-upload-dropzone"
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                    >

                        <div className="video-upload-icon">

                            <svg
                                viewBox="0 0 24 24"
                                fill="none"
                                strokeWidth="1.5"
                            >
                                <path d="M12 16V4M7.5 8.5 12 4l4.5 4.5M5 19h14"/>
                            </svg>

                        </div>


                        <h1>
                            Drop a rally video here
                        </h1>


                        <div className="video-upload-sub">
                            MP4, MOV, AVI — broadcast angle
                        </div>


                        <button
                            className="video-upload-browse"
                            onClick={() =>
                                document
                                    .getElementById("fileInput")
                                    .click()
                            }
                        >
                            Browse file
                        </button>


                        <input
                            id="fileInput"
                            className="video-upload-input"
                            type="file"
                            accept="video/mp4,video/quicktime,video/x-msvideo"
                            onChange={handleFileSelect}
                        />

                    </div>
                </div>
            )}
            


            <div className="video-upload-selected">
            </div>


            <div className="video-upload-features">

                <div className="video-upload-feature">
                    <strong>Ball tracking</strong>

                    <span>
                        ML-powered bounce
                        <br />
                        detection
                    </span>
                </div>


                <div className="video-upload-feature">
                    <strong>Shot mapping</strong>

                    <span>
                        Georeferenced court overlay
                    </span>
                </div>


                <div className="video-upload-feature">
                    <strong>Rally replay</strong>

                    <span>
                        Sequence-by-sequence view
                    </span>
                </div>

            </div>

        </section>
    );
}

export default VideoUpload;