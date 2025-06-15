import streamlit as st
import cv2
import numpy as np
from PIL import Image
from color_match import complete_color_matching_pipeline

st.title("Seamless Person Integration")

person_file = st.file_uploader("upload person image (.png with transparent background)", type=["png"])
bg_file = st.file_uploader("upload background image", type=["jpg", "png"])

if st.button("Generate Composite") and person_file and bg_file:
    with open("person_no_bg.png", "wb") as f:
        f.write(person_file.getbuffer())
    with open("bg.jpg", "wb") as f:
        f.write(bg_file.getbuffer())

    complete_color_matching_pipeline()

    st.image("final_composite.png", caption="Final Composite", use_column_width=True)
