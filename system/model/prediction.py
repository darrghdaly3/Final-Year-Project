import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import mediapipe as mp

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

## Adding a function to detect faces in the media to try increase accuracy and reduce background noise
def face_detection(image_path):
    
    ## Reusing code from the PoC and the extract_detect_crop.py scripts
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_detect = mp.solutions.face_detection
    with mp_detect.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as detector:
        results = detector.process(rgb)
    
    if not results.detections:
        return None
    
    detection = max(results.detections, key=lambda d: d.score[0])
    box = detection.location_data.relative_bounding_box     ## Creating the bounding box around the faces (cannot be seen)
    
    ## Convert MediaPipe locations into actual pixel coordinates
    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    face = image[y1:y2, x1:x2]
    
    if face.size == 0:
        return None
    
    ## Converting the bgr image to rgb for model to read
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    
    face_pil.show() ## Showing the cropped image to conform mediapipe worked (will be removed)
    
    return face_pil

## Classification and Confidence Score Function
def model_prediction(image_path):
    face = face_detection(image_path)

    if face is None:
        return "No Face Detected in Media!", 0.0
    
    ##image = Image.open(image_path).convert("RGB")
    image = image_transform(face).unsqueeze(0)
    
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
    
    image_path = "system/model/test.jpg"
    label, confidence = model_prediction(image_path)
    
    print("Classification:", label)
    print(f"Confidence Score: {round(confidence*100)}%")
    
