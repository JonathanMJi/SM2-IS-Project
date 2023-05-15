from markdown import markdown
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import preprocessing
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('model.h5')

def getResult(image):
    image = image.resize((64,64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = 0
    result = np.argmax(model.predict(input_img))
    return result

def get_className(classNo):
    Tumor_html="""
        <div style="background-color:#F08080;padding:10px">
        <h2 style="color:white;text-align:center;">Tumor detected</h2>
        </div>
        """
    Normal_html="""
        <div style="background-color:#5ec391 ;padding:10px">
        <h3 style="color:white;text-align:center;">Tumor not detected</h3>
        </div>
        """
    if classNo == 0:
        st.markdown(Normal_html, unsafe_allow_html=True)
    elif classNo == 1:
        st.markdown(Tumor_html, unsafe_allow_html=True)



def main():
    st.title("AI Brain Tumor Detection Tool")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Deep Learning Model</h2>
    <h5 style="color:white;text-align:center;">This model utilizes CNN (Convolutional Neural Networks) for brain cancer detection.</h5>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
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
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner("Working...."):               
                #plt.imshow(image)
                #plt.axis("off")
                predictions = getResult(image)
                get_className(predictions)

    help_html = """
    <div style="background-color:#025246 ;padding:20px">
    <h5 style="color:white;;text-align:center;">Info</h5>
    <p style="color:white">This deep learning web application utilizes Convolutional Neural to identify the presence of brain tumors from patient MRIs.</p>
    <p style="color:white">The dataset used to train this model is Ahmed Hamada's "Br35H :: Brain Tumor Detection 2020" on Kaggle.</p>
    </div>
    """
    st.markdown(help_html, unsafe_allow_html=True)

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")

                

if __name__ == "__main__":
    main()
