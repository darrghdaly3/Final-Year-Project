import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model_path = "system/model/efficientnet/best_trained_model.pt"

## Using the same Image Transform used for model training images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## Loading in the Trained Model
model = models.efficientnet_b0(
    weights = None
)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.load_state_dict(torch.load(model_path, map_location="cpu"))   ## Telling the model to run on the system cpu
model.eval()

## Classification and Confidence Score Function
def model_prediction(image_path):
    
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        
        probability = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probability, 1)
        
    if prediction.item() == 0:
        label = "Fake" 
    else:
        label = "Real"
        
    return label, confidence.item()

## Testing the model with a random image (Will be removed later)
if __name__ == "__main__":
    
    image_path = "system/model/sample.jpg"
    label, confidence = model_prediction(image_path)
    
    print("Classification:", label)
    print(f"Confidence Score: {round(confidence*100)}%")
    
