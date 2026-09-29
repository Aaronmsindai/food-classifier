# 🍕 Food Image Classifier

A PyTorch deep learning project that classifies images into 9 food categories using transfer learning.

## Results
- **Model:** ResNet18 (fine-tuned from ImageNet)
- **Validation Accuracy:** ~79%
- **Classes:** donuts, french_fries, hamburger, ice_cream, pizza, ramen, steak, sushi, tacos

## Project Structure
- scripts_download_food101.py — Streams a 9-class subset from HuggingFace Food-101
- train.py — Trains ResNet18 with data augmentation
- predict.py — Runs inference on new images

## How to Use
pip install -r requirements.txt
python scripts_download_food101.py
python train.py
python predict.py <path_to_image>

## Insights
The model performs strongly on visually distinct classes (ramen, tacos) but occasionally confuses visually similar classes (ice cream vs ramen, pizza vs tacos).
