import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import sys
import os

# 1. Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Rebuild the exact same architecture used for training
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)
model.load_state_dict(torch.load("food_classifier_resnet.pth", map_location=device))
model = model.to(device)
model.eval()

# 3. Same transforms as validation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 4. The class names (in alphabetical order, matching ImageFolder)
classes = ['donuts', 'french_fries', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi', 'tacos']

# 5. Pick an image to test on
# Default to a random val image if no path is provided
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    import random
    class_dir = random.choice(classes)
    img_file = random.choice(os.listdir(f"data/food101_subset/val/{class_dir}"))
    img_path = f"data/food101_subset/val/{class_dir}/{img_file}"
    print(f"No image provided. Testing on a random val image from '{class_dir}'")

# 6. Load and predict
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]
    predicted_idx = probabilities.argmax().item()

print(f"\nImage: {img_path}")
print(f"Predicted: {classes[predicted_idx]}")
print(f"Confidence: {probabilities[predicted_idx].item() * 100:.2f}%")
print("\nAll probabilities:")
for cls, prob in sorted(zip(classes, probabilities.tolist()), key=lambda x: -x[1]):
    print(f"  {cls:15s}: {prob*100:5.2f}%")
