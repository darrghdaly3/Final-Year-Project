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
    layout="wide"
)

def load_css(file_name):
    css_path = Path(__file__).parent / file_name
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")


## Naming the Web App and giving a description
st.markdown('<div class="main-title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-text">Upload an image or video you want to test</div>', unsafe_allow_html=True)

## Upload Area
center_left, center_mid, center_right = st.columns([1, 2, 1])
with center_mid:
    upload_area = st.file_uploader(
        "Upload an image or video (.jpg .jpeg .png .mp4)",
        type=["jpg", "jpeg", "png", "mp4"]
    )

if upload_area is None:
    st.info("Please upload an image or video file to begin detection!")

else:
    st.session_state.upload_area = upload_area
    file_extension = upload_area.name.split(".")[-1].lower()

    ## Save Uploaded File Temporarily
    upload_area.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(upload_area.read())
        temp_path = tmp_file.name

    media_col, results_col = st.columns(2)
    
    ## Image detection pipeline
    if file_extension in ["jpg", "jpeg", "png"]:
    
        with media_col:
            st.subheader("Uploaded Image")
            try:
                upload_area.seek(0)
                uploaded_img = Image.open(upload_area)
                left, center, right = st.columns([1,2,1])
                with center:
                    st.image(uploaded_img, width=350)
            except Exception as e:
                st.error(f"Could not display uploaded image: {e}")

        with results_col:
            try:
                face, label, confidence = model_prediction(temp_path)

                if face is None:
                    st.error("No face detected. Please upload media containing one or more faces.")
                else:
                    st.subheader("Results")

                    if label == "Fake":
                        st.markdown('<div class="result-fake">Fake</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-real">Real</div>', unsafe_allow_html=True)

                    st.write("Confidence Score:", f"{round(confidence * 100)}%")

                    if label == "Fake":
                        gradcam, heatmap = create_gradcam(face)
                        selected_area = detect_features(heatmap)

                        st.write(
                            "Explanation:",
                            f"The detector focused mainly on the {selected_area} area(s), which are possibly manipulated or fake features."
                        )

                        gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                        left, center, right = st.columns([1,2,1])
                        with center:
                            st.image(gradcam_rgb, caption="Grad-CAM Heatmap", width=350)

                    elif label == "Real" and confidence < 0.80:
                        gradcam, heatmap = create_gradcam(face)
                        selected_area = detect_features(heatmap)

                        st.write(
                            "Explanation:",
                            f"The detector found the media to be real, but is low in confidence due to the {selected_area} area(s)."
                        )

                        gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                        left, center, right = st.columns([1,2,1])
                        with center:
                            st.image(gradcam_rgb, caption="Grad-CAM Heatmap", width=350)

            except Exception as e:
                st.error(f"An error occurred while processing the image: {e}")
    
    ## Video Detection Pipeline
    elif file_extension in ["mp4"]:
    
        with media_col:
            st.subheader("Uploaded Video")
            st.video(temp_path)

        with results_col:
            try:
                result = predict_video(temp_path, frame_checks=10)

                if not result["success"]:
                    st.error(result["error"])
                else:
                    st.subheader("Results")

                    if result["label"] == "Fake":
                        st.markdown('<div class="result-fake">Fake</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-real">Real</div>', unsafe_allow_html=True)

                    st.write("Confidence Score:", f"{round(result['confidence'] * 100)}%")

                    if result["label"] == "Fake":
                        gradcam, heatmap = create_gradcam(result["best_face"])
                        selected_area = detect_features(heatmap)

                        st.write(
                            "Explanation:",
                            f"The detector focused mainly on the {selected_area} area(s), which are possibly manipulated or fake features."
                        )

                        gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                        left, center, right = st.columns([1,2,1])
                        with center:
                            st.image(gradcam_rgb, caption="Grad-CAM Heatmap", width=350)

                    elif result["label"] == "Real" and result["confidence"] < 0.80:
                        gradcam, heatmap = create_gradcam(result["best_face"])
                        selected_area = detect_features(heatmap)

                        st.write(
                            "Explanation:",
                            f"The detector found the media to be real, but is low in confidence due to the {selected_area} area(s)."
                        )

                        gradcam_rgb = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
                        left, center, right = st.columns([1,2,1])
                        with center:
                            st.image(gradcam_rgb, caption="Grad-CAM Heatmap", width=350)

            except Exception as e:
                st.error(f"An error occurred while processing the video: {e}")

    else:
        st.error("Unsupported file type.")
