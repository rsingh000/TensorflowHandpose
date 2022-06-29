// Import dependencies
import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
import { nextFrame } from "@tensorflow/tfjs";
// 2. TODO - Import drawing utility here
import {drawRect} from "./utilities"; 

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Main function
  const runCoco = async () => {
    // 3. TODO - Load network 
    const net = await tf.loadGraphModel('https://storage.googleapis.com/directiontfod/model.json')
    
    // Loop and detect hands
    setInterval(() => {
      detect(net);
    }, 16.7);
  };

  const detect = async (net) => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // 4. TODO - Make Detections
      const img = tf.browser.fromPixels(video)
      const resized = tf.image.resizeBilinear(img, [640,480])
      const casted = resized.cast('int32')
      const expanded = casted.expandDims(0)
      const obj = await net.executeAsync(expanded)
      
      console.log(await obj[2].array());

      const boxes = await obj[7].array()
      const classes = await obj[5].array()
      const scores = await obj[2].array()
    
      // Draw mesh
      const ctx = canvasRef.current.getContext("2d");

      // 5. TODO - Update drawing utility
      // drawSomething(obj, ctx)  
      requestAnimationFrame(()=>{drawRect(boxes[0], classes[0], scores[0], 0.7, videoWidth, videoHeight, ctx)}); 

      tf.dispose(img)
      tf.dispose(resized)
      tf.dispose(casted)
      tf.dispose(expanded)
      tf.dispose(obj)

    }
  };

  useEffect(()=>{runCoco()},[]);

  return (
    <div className="App">
      <div className="guidelines">
        <h2>Some Important Instructions <br />for better performance of <br /> machine learning model.</h2>
        <div className="list">
          <span>Hold your hand still for couple of seconds until bounding boxes seems to appear.</span>
          <span>Ensure the enough amount of light in the room. This model has not been trained in darkness.</span>
        </div>
      </div>
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          muted={true} 
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: '100%',
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 8,
            width: '100%',
            height: 480,
          }}
        />
      </header>
      <h2>Take a look at few tutorials below</h2>
      <div className="examples">
          <figure className="up">
            <img src="https://i.ibb.co/VqZC04H/up.jpg"
                alt="Up Direction" />
            <figcaption>UP handpose detected at the precision 91% with blue outbound box</figcaption>
          </figure>
          <figure className="down">
            <img src="https://i.ibb.co/99QwqH9/down.jpg"
                alt="Down Direction" />
            <figcaption>DOWN handpose detected at the precision 97% with green outbound box</figcaption>
          </figure>
          <figure className="left">
            <img src="https://i.ibb.co/HFJxLPj/left.jpg"
                alt="Left Direction" />
            <figcaption>LEFT handpose detected at the precision 86% with red outbound box</figcaption>
          </figure>
          <figure className="right">
            <img src="https://i.ibb.co/dmSNsJz/right.jpg"
                alt="Right Direction" />
            <figcaption>RIGHT handpose detected at the precision 98% with yellow outbound box</figcaption>
          </figure>
      </div>
    </div>
  );
}

export default App;
