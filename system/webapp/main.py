import streamlit as st
import os
import tempfile
import cv2
from PIL import Image

from system.model.prediction import model_prediction
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

