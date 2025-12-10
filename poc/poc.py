import cv2
import mediapipe as mp

## Load in the stored image for OpenCV to read
image_path = "poc/images/sample.jpg"
image = cv2.imread(image_path)

## Colour Conversion from BGR to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

mp_detect = mp.solutions.face_detection

## MediaPipe creates the Face Detector and stores the area the face is found
with mp_detect.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as detector:
    results = detector.process(rgb)
    
if results.detections:
    h, w, _ = image.shape
    for detection in results.detections:
        box = detection.location_data.relative_bounding_box
        
        ## Convert MediaPipe locations into actual pixel coordinates
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)
        
        ## Draw box around face found
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
## Display Image with Box
cv2.imshow("Face Detected", image)
cv2.waitKey(0)
