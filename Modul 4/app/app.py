import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Chihuahua vs Muffin Classifier",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('CNN_Prak4_ML.h5')

def preprocess_image(img):
    img = img.resize((128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

LABEL_CLASS = {
    0: "chihuahua",
    1: "muffin",
}

def main():
    st.title("Chihuahua vs Muffin Classifier")
    st.write("Upload an image and the model will predict whether it's a chihuahua or a muffin!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            model = load_model()
            
            processed_image = preprocess_image(image)
            
            with st.spinner('Predicting...'):
                prediction = model.predict(processed_image)
                pred_class = LABEL_CLASS[np.argmax(prediction)]
                confidence = float(prediction.max()) * 100
            
            st.success(f'Prediction: {pred_class.upper()}')
            st.info(f'Confidence: {confidence:.2f}%')
            
            st.write("Class Probabilities:")
            for i, prob in enumerate(prediction[0]):
                st.progress(float(prob))
                st.write(f"{LABEL_CLASS[i]}: {float(prob)*100:.2f}%")

if __name__ == "__main__":
    main()