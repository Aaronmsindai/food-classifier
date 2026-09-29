import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Loading with Augmentation (improves generalization)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder(root='data/food101_subset/train', transform=train_transform)
val_data = datasets.ImageFolder(root='data/food101_subset/val', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(f"Classes: {train_data.classes}")
print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

# 3. Initialize Pretrained ResNet18 Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Replace the final layer to output 9 classes instead of 1000 ImageNet classes
model.fc = nn.Linear(model.fc.in_features, 9)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Use a lower learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4. Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

    # 5. Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Epoch [{epoch+1}/{epochs}] Validation Accuracy: {100 * correct / total:.2f}%\n")

print("Training complete!")

# Save the trained model weights
torch.save(model.state_dict(), "food_classifier_resnet.pth")
print("Model weights saved to food_classifier_resnet.pth")
