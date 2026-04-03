import streamlit as st

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