# 🍕 Food Image Classifier - Deep Learning MLOps Pipeline

A production-grade PyTorch deep learning pipeline that classifies images into 9 food categories using transfer learning. Features a fully decoupled microservice architecture with a Streamlit frontend and a containerized FastAPI backend.

## 🌐 Live Demos
🚀 **[Test the Live Web App (Streamlit)](https://food-classifier-rpb3l9d9gpvkdv82rzepkc.streamlit.app)**
⚙️ **[View the Live API Docs (Render)](https://food-classifier-api-2ux2.onrender.com/docs)**

*(Note: The Render API runs on a free tier. The first prediction may take ~30 seconds to wake up the server).*

## 🏗️ Architecture
1. **ML Backend (FastAPI + PyTorch):** A containerized REST API that accepts image uploads, processes them through the ResNet18 model, and returns predictions. Deployed via Docker on Render.
2. **Frontend (Streamlit):** An interactive UI hosted on Streamlit Community Cloud. It handles image uploads and sends asynchronous HTTP requests to the FastAPI backend.

## 🧠 Model Details
- **Architecture:** ResNet18 (fine-tuned from ImageNet using Transfer Learning)
- **Validation Accuracy:** ~79%
- **Classes:** donuts, french_fries, hamburger, ice_cream, pizza, ramen, steak, sushi, tacos

## 📂 Project Structure
```text
food-classifier/
├── api.py                  # FastAPI backend for model inference
├── app.py                  # Streamlit frontend UI
├── Dockerfile.api          # Docker configuration for the FastAPI backend
├── train.py                # Model training script
├── predict.py              # Local CLI inference script
├── scripts_download_food101.py # Script to stream dataset from HuggingFace
├── food_classifier_resnet.pth  # Saved model weights
└── requirements.txt        # Frontend dependencies
How to Run Locally (Full Microservice)

If you want to run the entire pipeline on your machine:

1. Start the FastAPI Backend:

```bash
pip install fastapi uvicorn torch torchvision pillow
uvicorn api:app --port 8000
```

2. Start the Streamlit Frontend (in a new terminal):

```bash
pip install streamlit requests pillow
streamlit run app.py


📊 Insights

The model performs strongly on visually distinct classes (ramen, tacos) but occasionally confuses visually similar classes (ice cream vs ramen, pizza vs tacos).

🛠️ Tech Stack

· Deep Learning: PyTorch, torchvision, Transfer Learning (ResNet18)
· Backend API: FastAPI, Uvicorn, Docker
· Frontend UI: Streamlit, requests
· Deployment: Render (API), Streamlit Cloud (UI)
