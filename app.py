import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# --- 1. Page Config ---
st.set_page_config(page_title="Food Classifier", page_icon="🍕", layout="centered")
st.title("🍕 AI Food Classifier")
st.markdown("Upload a photo of food, and my ResNet18 model will identify it. Trained on a 9-class subset of Food-101.")

# --- 2. Load Model (Cached so it only loads once) ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Streamlit Cloud uses CPU
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 9)
    # Load the saved weights
    model.load_state_dict(torch.load("food_classifier_resnet.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# --- 3. Define Transforms and Classes ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ['donuts', 'french_fries', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi', 'tacos']

# --- 4. File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("### Analyzing...")
    
    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
    
    # Display results
    st.success(f"**Prediction: {classes[predicted_idx].replace('_', ' ').title()}**")
    st.metric("Confidence", f"{probabilities[predicted_idx].item() * 100:.2f}%")
    
    # Show all probabilities as a bar chart
    st.write("### All Probabilities")
    probs_dict = {cls.replace('_', ' ').title(): prob.item() * 100 for cls, prob in zip(classes, probabilities)}
    st.bar_chart(probs_dict)

st.divider()
st.caption("Built with PyTorch, Transfer Learning, and Streamlit.")
