from markdown import markdown
import streamlit as st
import numpy as np
np.object = object

from PIL import Image
import tensorflow as tf
from keras.utils.np_utils import normalize

model = tf.keras.models.load_model('model.h5')

def detect(mri):
    mri=mri.resize((64,64))
    mri = np.array_(mri)
    mri = [mri]
    mri = normalize(mri, axis = 1)
    result =model.predict(mri).ravel()
    if result > 0.8:
        notumor="""
        <div style="background-color:#5ec391 ;padding:10px">
        <h3 style="color:white;text-align:center;">NO TUMOR DETECTED</h3>
        </div>
        """
        st.markdown(notumor, unsafe_allow_html=True)
    else:
        tumor="""
        <div style="background-color:#F08080;padding:10px">
        <h2 style="color:white;text-align:center;">TUMOR DETECTED</h2>
        </div>
        """
        st.markdown(tumor, unsafe_allow_html=True)

def main():
    st.title("AI Brain Tumor Detection")
    mri = st.file_uploader("", type=["png","jpg","jpeg"])
    button = f"""
    <style>
    div.stButton > button:first-child {{ background-color: #04AA6D;color: white; padding: 12px 20px;border: none;border-radius: 4px;cursor: pointer; }}
    <style>
    """
    st.markdown(button, unsafe_allow_html=True)
    class_btn = st.button("PREDICT")
    
    if mri is not None:
        image= Image.open(mri)
        st.image(image)

    if class_btn:
        if mri is None:
            st.write("INVALID SUBMISSION")
        else:
                pred = detect(image)
    st.image([Image.open('cm.png'),Image.open('roc.png')])
    def local_css(file):
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style/style.css")

if __name__ == "__main__":
    main()
