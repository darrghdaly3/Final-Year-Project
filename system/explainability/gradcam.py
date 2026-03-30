import torch
from torchvision import transforms
import cv2
import numpy as np
from system.model.prediction import model, model_prediction
from system.explainability.textual import detect_features 
import matplotlib.pyplot as plt

## Using the same Image Transform used for model training images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

activations = None
gradients = None

## Creates the forward hook that saves the features from the last convolutional layer
def get_features(module, input, output):
    global activations
    activations = output

## Creates the backward hook that saves the gradients from the last convolutional layer
def get_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

## Choosing the last convolutional layer of the EfficientNet model, and attaching the forward and backward hooks
chosen_layer = model.features[-1]
chosen_layer.register_forward_hook(get_features)
chosen_layer.register_full_backward_hook(get_gradients)

## The function that creates the Grad Cam heat map and places it over a face
def create_gradcam(face):
    image = image_transform(face).unsqueeze(0)
    image.requires_grad_()
    
    ## Forward Pass - Putting the image in the model and getting the clasification
    output = model(image)
    prediction = output.argmax(dim=1).item()
    
    ## Backward Pass - Tells what part of the image influenced the classification
    model.zero_grad()
    output[0, prediction].backward()
    
    ## Creating the Heatmap
    saved_gradients = gradients.mean([0, 2, 3])
    maps = activations[0].clone()
    
    for m in range(maps.shape[0]):
        maps[m] *= saved_gradients[m]
    
    ## Averaging all feature maps to create 1 map
    heatmap_raw = maps.mean(dim=0).detach().numpy()
    
    heatmap_raw = np.maximum(heatmap_raw, 0)
    heatmap_raw /= heatmap_raw.max() + 1e-8
    
    ## Resizing map and coverting colour so cv2 can dsiplay it
    heatmap_show = cv2.resize(heatmap_raw, (224, 224))
    
    heatmap_show = np.uint8(255 * heatmap_show)
    heatmap_show = cv2.applyColorMap(heatmap_show, cv2.COLORMAP_JET)
    
    ## Overlaying the heatmap over faces
    face_np = np.array(face.resize((224, 224)))
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    
    ## Blending the face and the heatmap
    map_layer = cv2.addWeighted(face_bgr, 0.6, heatmap_show, 0.4, 0)
    
    return map_layer, heatmap_raw

## Testing the heatmap on the image used in prediction.py
if __name__ == "__main__":
    
    image_path = "system/model/test.jpg"
    face, label, confidence = model_prediction(image_path)
    
    if face is None:
        print("No Face Detected")
    else:
        gradcam, heatmap = create_gradcam(face)
        selected_area = detect_features(heatmap)

        # Show result
        print("Classification:", label)
        print(f"Confidence Score: {round(confidence*100)}%")
        
        if label == "Fake":
            print("Explanation:", f"The detector focused mainly on the {selected_area} area(s), which are possdibly manipulated or fake features.")

            plt.imshow(cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB), alpha=0.4)
            plt.axis("off")
            plt.show()

        elif label == "Real" and confidence < 0.80:
            print("Explanation:", f"The detector found the media to be real, but is low in confidence due to the {selected_area} area(s).")

            plt.imshow(cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB), alpha=0.4)
            plt.axis("off")
            plt.show()