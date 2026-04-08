import cv2
from system.model.prediction import model_prediction

## Predict if a video is real or fake by chacking every tenth frame
def predict_video(video_path, frame_checks = 10):
    cap = cv2.VideoCapture(video_path)
    
    print("Video path:", video_path)
    print("Video opened:", cap.isOpened())

    if not cap.isOpened():
        return {
            "success": False,
            "error": "Could not open video. Check the files format is mp4!"
        }
    
    
    frame_index = 0
    frames_processed = 0
    real_count = 0
    fake_count = 0
    total_confidence = 0

    ## Varaiables that store the frame where the model is most confident, real or fake
    best_face = None
    best_confidence = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_index % frame_checks == 0:
            frame_path = "video_frame.jpg"
            saved = cv2.imwrite(frame_path, frame)
            
            if not saved:
                frame_index += 1
                continue

            ## Reusing the pipeline for predicting images
            face, label, confidence = model_prediction(frame_path)

            if face is not None:
                frames_processed += 1
                total_confidence += confidence

                ## Counting predictions, will determin the models final classification of the video
                if label == "Real":
                    real_count += 1
                else:
                    fake_count += 1

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_face = face

        frame_index += 1

    cap.release()

    if frames_processed == 0:
        return {
            "success": False,
            "error": "No face detected in the video."
        }

    if fake_count > real_count:
        final_label = "Fake"
    else:
        final_label = "Real"

    average_confidence = total_confidence / frames_processed

    return {
        "success": True,
        "label": final_label,
        "confidence": average_confidence,
        "frames_processed": frames_processed,
        "real_frames": real_count,
        "fake_frames": fake_count,
        "best_face": best_face
    }

## Testing the code file with a set video
if __name__ == "__main__":
    video_path = "system/model/real_0002.mp4"
    result = predict_video(video_path)

    if result["success"]:
        print("Classification:", result["label"])
        print(f"Confidence Score: {round(result['confidence'] * 100)}%")
        print("Frames Processed:", result["frames_processed"])
        print("Real Frames:", result["real_frames"])
        print("Fake Frames:", result["fake_frames"])
        
        if result["best_face"] is not None:
            result["best_face"].show()
            
    else:
        print("Error:", result["error"])