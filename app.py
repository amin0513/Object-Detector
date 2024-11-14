from ultralytics import YOLO
import streamlit as st
from PIL import Image


st.title("Basic Object Detection")
st.write("Implement a simple object detection system that can locate and identify objects in images.")
st.write("Upload an image please")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    model = YOLO('yolov8l.pt')
    results = model(image)
    for result in results:
        st.write("Detected objects:")
        detected_image = result.plot()
        st.image(detected_image, caption="Detected Objects", use_column_width=True)        
else:
    st.write("Please upload an image file.")
