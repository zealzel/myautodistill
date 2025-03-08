import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";

const FACING_MODE_USER = "user";
const FACING_MODE_ENVIRONMENT = "environment";
const WIDTH = 640;
const HEIGHT = 480;
// const WIDTH = 1280;
// const HEIGHT = 760;
const RECOG_WIDTH = 320;
const RECOG_HEIGHT = 240;

function App() {
  const webcamRef = useRef(null);
  const [recognizedFaces, setRecognizedFaces] = useState([]);
  const [facingMode, setFacingMode] = React.useState(FACING_MODE_USER);

  const capture = React.useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot({
      width: RECOG_WIDTH,
      height: RECOG_HEIGHT,
    });

    try {
      const response = await fetch("/detect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageSrc,
          width: RECOG_WIDTH,
          height: RECOG_HEIGHT,
        }),
      });
      const data = await response.json();
      setRecognizedFaces(data.detectedObjects || []);
    } catch (error) {
      console.error("Error detecting objects:", error);
    }
  }, [webcamRef]);

  let videoConstraints = {
    facingMode: facingMode,
    width: WIDTH,
    height: HEIGHT,
  };

  const handleClick = React.useCallback(() => {
    setFacingMode((prevState) =>
      prevState === FACING_MODE_USER
        ? FACING_MODE_ENVIRONMENT
        : FACING_MODE_USER,
    );
  }, []);

  useEffect(() => {
    const intervalId = setInterval(() => {
      capture();
    }, 200);
    return () => clearInterval(intervalId);
  }, [capture]);

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
      />
      <button onClick={handleClick}>Switch camera</button>
      {recognizedFaces.map((object, index) => {
        const video = webcamRef.current && webcamRef.current.video;
        const previewWidth = video ? video.clientWidth : WIDTH;
        const previewHeight = video ? video.clientHeight : HEIGHT;
        const scaleX = previewWidth / RECOG_WIDTH;
        const scaleY = previewHeight / RECOG_HEIGHT;

        const top = object.location.top * scaleY;
        const left = object.location.left * scaleX;
        const boxWidth =
          (object.location.right - object.location.left) * scaleX;
        const boxHeight =
          (object.location.bottom - object.location.top) * scaleY;

        // 根據物體類型設定不同的邊框顏色
        const borderColor = object.label.includes("Hand")
          ? "#00ff00"
          : "#ff0000";

        return (
          <div
            key={index}
            style={{
              position: "absolute",
              border: `2px solid ${borderColor}`,
              top: `${top}px`,
              left: `${left}px`,
              width: `${boxWidth}px`,
              height: `${boxHeight}px`,
              color: "white",
              backgroundColor: "rgba(0, 0, 0, 0.5)",
              fontSize: "14px",
              fontWeight: "bold",
              textAlign: "center",
              lineHeight: "1",
              padding: "2px",
            }}
          >
            {object.label}
          </div>
        );
      })}
    </div>
  );
}

export default App;
