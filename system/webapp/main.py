import streamlit as st
import os
import tempfile
import cv2
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from system.model.prediction import model_prediction
from system.model.video_prediction import predict_video
from system.explainability.gradcam import create_gradcam
from system.explainability.textual import detect_features

## Creating the Page
st.set_page_config(
    page_title="Deepfake Detection System",
    layout="centered"
)

## Naming the Web App and giving a description
st.title("Deepfake Detection System")
st.write(
    "Upload an image or video you want to test"
)

## Upload Area
upload_area = st.file_uploader(
    "Upload an image or video (.jpg .jpeg .png .mp4)",
    type=["jpg", "jpeg", "png", "mp4"]
)

if upload_area is None:
    st.info("Please upload an image or video file to begin detection!")

else:
    file_extension = upload_area.name.split(".")[-1].lower()

    ## Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(upload_area.read())
        temp_path = tmp_file.name
    
    ## Image detection pipeline
    if file_extension in ["jpg", "jpeg", "png"]:
        st.subheader("Uploaded Image")
        st.image(temp_path, use_container_width=True)

        try:
            face, label, confidence = model_prediction(temp_path)
            
            if face is None:
                st.error("No face detected. Please upload media containing one or more faces.")
            else:
                st.subheader("Results")
                st.write("Classification:", label)
                st.write("Confidence Score:", f"{round(confidence * 100)}%")
                
                ## Reusing the code from gradcam.py for displaying heatmaps and explanations
                if label == "Fake":
                    gradcam, heatmap = create_gradcam(face)
                    selected_area = detect_features(heatmap)

                    st.write(
                        "Explanation:", f"The detector focused mainly on the {selected_area} area(s), which are possibly manipulated or fake features."
                    )

                    gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                    st.image(gradcam_rgb, caption="Grad-CAM Heatmap", use_container_width=True)
                    
                elif label == "Real" and confidence < 0.80:
                    gradcam, heatmap = create_gradcam(face)
                    selected_area = detect_features(heatmap)

                    st.write(
                        "Explanation:", f"The detector found the media to be real, but is low in confidence due to the {selected_area} area(s)."
                    )

                    gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                    st.image(gradcam_rgb, caption="Grad-CAM Heatmap", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
    
    ## Video Detection Pipeline
    elif file_extension in ["mp4"]:
        st.subheader("Uploaded Video")
        st.video(temp_path)

        try:
            result = predict_video(temp_path, frame_checks=10)

            if not result["success"]:
                st.error(result["error"])
            else:
                st.subheader("Results")
                st.write("Classification:", result["label"])
                st.write("Confidence Score:", f"{round(result['confidence'] * 100)}%")

                if result["label"] == "Fake":
                    gradcam, heatmap = create_gradcam(result["best_face"])
                    selected_area = detect_features(heatmap)

                    st.write(
                        "Explanation:", f"The detector focused mainly on the {selected_area} area(s), which are possibly manipulated or fake features."
                    )

                    gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                    st.image(gradcam_rgb, caption="Grad-CAM Heatmap", use_container_width=True)

                elif result["label"] == "Real" and result["confidence"] < 0.80:
                    gradcam, heatmap = create_gradcam(result["best_face"])
                    selected_area = detect_features(heatmap)

                    st.write(
                        "Explanation:", f"The detector found the media to be real, but is low in confidence due to the {selected_area} area(s)."
                    )

                    gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                    st.image(gradcam_rgb, caption="Grad-CAM Heatmap", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing the video: {e}")

    else:
        st.error("Unsupported file type.")
