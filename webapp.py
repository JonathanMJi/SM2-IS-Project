from markdown import markdown
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils.np_utils import normalize

model = tf.keras.models.load_model('model.h5')

def detect(image):
    img = image
    img=img.resize((64,64))
    img = np.array(img)
    img = [img]
    img = normalize(img, axis = 1)
    result =model.predict(img).ravel()
    if result > 0.8:
        Normal_html="""
        <div style="background-color:#5ec391 ;padding:10px">
        <h3 style="color:white;text-align:center;">Tumor not detected</h3>
        </div>
        """
        st.markdown(Normal_html, unsafe_allow_html=True)
    else:
        Tumor_html="""
        <div style="background-color:#F08080;padding:10px">
        <h2 style="color:white;text-align:center;">Tumor detected</h2>
        </div>
        """
        st.markdown(Tumor_html, unsafe_allow_html=True)

def main():
    st.title("SM2 IS Project: AI Brain Tumor Detection")
    uploaded_file = st.file_uploader("", type=["png","jpg","jpeg"])
    s = f"""
    <style>
    div.stButton > button:first-child {{ background-color: #04AA6D;color: white; padding: 12px 20px;border: none;border-radius: 4px;cursor: pointer; }}
    <style>
    """
    st.markdown(s, unsafe_allow_html=True)
    class_btn = st.button("Predict")
    
    if uploaded_file is not None:
        image= Image.open(uploaded_file)
        st.image(image)

    if class_btn:
        if uploaded_file is None:
            st.write("Invalid File Type")
        else:
                pred = detect(image)
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style/style.css")

if __name__ == "__main__":
    main()
