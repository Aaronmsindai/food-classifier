from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI(title="Food Classifier API")

# 1. Load Model globally on startup
device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

# We assume the .pth file is in the root directory
model.load_state_dict(torch.load("food_classifier_resnet.pth", map_location=device))
model = model.to(device)
model.eval()

# 2. Define Transforms and Classes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ['donuts', 'french_fries', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi', 'tacos']

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "ResNet18-Food101"}

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
            
        # Format output
        result = {
            "prediction": classes[predicted_idx].replace('_', ' ').title(),
            "confidence": round(probabilities[predicted_idx].item() * 100, 2),
            "all_probabilities": {
                cls.replace('_', ' ').title(): round(prob.item() * 100, 2) 
                for cls, prob in zip(classes, probabilities)
            }
        }
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
