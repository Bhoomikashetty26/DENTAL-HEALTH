import streamlit as st
import torch
from PIL import Image
import numpy as np
import pathlib

# Permanently replace PosixPath with WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

from internal.detection import detect_disease
from internal.image_processing import process_image
from internal.rendering import draw_boxes

# Load models
xray_model = torch.hub.load(
    './yolov5', 'custom', path='models/xray.pt', force_reload=True, source="local"
)
camera_model = torch.hub.load(
    './yolov5', 'custom', path='models/camera.pt', force_reload=True, source="local"
)

# Streamlit app layout
st.title("Dental Disease Detection App")
st.write("Upload an X-ray or Camera image to detect dental diseases.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model_type = st.selectbox("Select Model", ("xray", "camera"))

    image_array = np.array(image)

    if st.button("Detect Disease"):
        model = xray_model if model_type == 'xray' else camera_model
        detections = detect_disease(image_array, model)
        image_with_boxes = draw_boxes(image_array, detections, model.names)
        output_image = Image.fromarray(image_with_boxes)
        st.image(output_image, caption="Detection Results", use_container_width=True)