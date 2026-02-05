import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm 

## Paths to where the split videos are stored and path where the cropped frames will be stored
split_path = Path("system/data/splitvids")
frame_path = Path("system/data/processed")

## Setting were the frames will be taken from in each video and the size they must be set to
frame_positions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
image_size = (224, 224)

## Function to extract 4 frames fom each video at the specified positions above
def frame_extraction(cap):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = sorted(set(int(p * (total_frames - 1)) for p in frame_positions))
    
    frames = []     ## Creating a list to store the exracted frames in
    for id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, id)
        ok, frame = cap.read()
        
        if ok:
            frames.append((id, frame))
        
    return frames

## Function to detect and crop faces within the frames extracted from above
def face_detection(frame, detector):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = detector.process(rgb)
    
    faces = []      ## Creating a list to store the found faces in
    if not results.detections:
        return faces
    
    detection = max(results.detections, key=lambda d: d.score[0])
    box = detection.location_data.relative_bounding_box     ## Creating the bounding box around the faces (cannot be seen)
    
    ## Convert MediaPipe locations into actual pixel coordinates
    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)
    
    ## Resizing the faces cropped and adding them to the faces list
    face = frame[y1:y2, x1:x2]
    if face.size != 0:
        face = cv2.resize(face, image_size)
        faces.append(face)
    
    return faces

## Function to actually process a single video usign the functions above together
def face_extraction(video_path, split, cls, detector):
    
    ## Selecting the videos and calling the frame extraction on them
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    
    frames = frame_extraction(cap)
    cap.release()
    
    processed_data = frame_path / split / cls
    base_name = video_path.stem
    
    ## Detecting faces, cropping them and saving them as jpg files
    for frame_id, frame in frames:
        faces = face_detection(frame, detector)
        for i, face in enumerate(faces):
            frame_store = processed_data / f"{base_name}_f{frame_id}_face{i}.jpg"
            cv2.imwrite(str(frame_store), face)
            
## Function to create the face detector model and to call each video individually for processing
def main():
    ## Creating the face detection model
    mp_detect = mp.solutions.face_detection
    with mp_detect.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as detector:
        
        ## Looping through each video in the folders
        for split in ["train", "validate", "test"]:
            for cls in ["real", "fake"]:
                videos = list((split_path / split / cls).glob("*.mp4"))
                
                ## Calling the face extraction function so the videos can be processed
                for video in tqdm(videos, desc=f"{split}/{cls}"):
                    face_extraction(video, split, cls, detector)
                    
    print("Frames Successfully Extracted and Stored!")

if __name__ == "__main__":
    main()
    