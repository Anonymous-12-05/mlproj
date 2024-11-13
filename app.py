import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from PIL import Image

# Load YOLO model for pose detection
model = YOLO('yolov8m-pose.pt')

# Streamlit app setup
st.title("Real-time Pose Detection using YOLOv8")
st.write("This app uses YOLOv8 to perform pose detection in real-time through your webcam.")

# Initialize video capture and create a placeholder for video feed
run = st.checkbox('Open Camera')
frame_placeholder = st.empty()

# Start video capture
if run:
    cap = cv2.VideoCapture(0)

    # Stream video and apply YOLO model
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Apply YOLO model on the frame
        results = model(frame, conf=0.3)
        frame = results[0].plot()  # Draw keypoints on the frame

        # Display frame with Streamlit
        frame_placeholder.image(frame, channels="BGR")

    # Release the camera
    cap.release()

else:
    st.write("Click 'Open Camera' to start the video stream.")
