import streamlit as st
import requests
from PIL import Image

# --- 1. Page Config ---
st.set_page_config(page_title="Food Classifier", page_icon="🍕", layout="centered")
st.title("🍕 AI Food Classifier")
st.markdown("Upload a photo of food, and the API will identify it. Trained on a 9-class subset of Food-101.")

API_URL = "https://food-classifier-api-2ux2.onrender.com/predict"

# --- 3. File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
    
    st.write("### Analyzing...")
    
    if st.button("Run Prediction", type="primary"):
        with st.spinner("Sending image to API..."):
            try:
                # Prepare the file to send
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send to FastAPI
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                data = response.json()
                
                # Display results
                st.success(f"**Prediction: {data['prediction']}**")
                st.metric("Confidence", f"{data['confidence']}%")
                
                # Show all probabilities
                st.write("### All Probabilities")
                st.bar_chart(data['all_probabilities'])
                
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

st.divider()
st.caption("Built with PyTorch, FastAPI, Transfer Learning, and Streamlit.")
